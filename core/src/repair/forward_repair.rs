//! Bounded discovery of successors while orphan repair walks back from newer slots.
use {
    super::{
        repair_service::{REPAIR_REQUEST_TIMEOUT_MS, RepairService},
        serve_repair::ShredRepairType,
    },
    solana_clock::Slot,
    solana_ledger::{blockstore::Blockstore, blockstore_db::DBPinnableSlice},
    solana_runtime::bank_forks::BankForks,
    std::{
        collections::{BTreeMap, BTreeSet, HashMap},
        ops::Bound::{Excluded, Unbounded},
        time::{Duration, Instant},
    },
};

const PROBES_PER_CHECK: usize = 8;
const LOOKAHEAD: Slot = 32;
const CHECK_INTERVAL: Duration = Duration::from_millis(REPAIR_REQUEST_TIMEOUT_MS);
const CHILD_STALL_INTERVAL: Duration = Duration::from_secs(1);

struct Frontier {
    next_offset: Slot,
    // consumed/received are cheap progress hints, not a count of inserted shreds.
    children: Vec<(Slot, u64, u64)>,
    last_progress: Instant,
}

impl Frontier {
    fn new(now: Instant) -> Self {
        Self {
            next_offset: 1,
            children: Vec::new(),
            last_progress: now,
        }
    }

    fn needs_discovery<'db>(
        &mut self,
        blockstore: &'db Blockstore,
        pinnable_slice: &mut DBPinnableSlice<'db>,
        anchor: Slot,
        now: Instant,
    ) -> bool {
        if blockstore.is_dead(anchor) {
            return false;
        }
        let mut children = Vec::new();
        match blockstore.meta_repair_into(anchor, pinnable_slice) {
            Ok(Some(meta)) => {
                for child in meta.next_slots {
                    if blockstore.is_dead(child) {
                        continue;
                    }
                    match blockstore.meta_repair_into(child, pinnable_slice) {
                        Ok(Some(meta)) => children.push((child, meta.consumed, meta.received)),
                        Ok(None) => (),
                        Err(_) => return false,
                    }
                }
            }
            Ok(None) => (), // A snapshot's frozen bank need not have metadata.
            Err(_) => return false,
        }
        children.sort_unstable();
        if children != self.children {
            self.children = children;
            self.last_progress = now;
        }
        self.children.is_empty() || now.duration_since(self.last_progress) >= CHILD_STALL_INTERVAL
    }

    fn eligible<'db>(
        &mut self,
        blockstore: &'db Blockstore,
        pinnable_slice: &mut DBPinnableSlice<'db>,
        anchor: Slot,
        highest_orphan: Option<Slot>,
        now: Instant,
    ) -> bool {
        anchor
            .checked_add(LOOKAHEAD)
            .is_some_and(|end| highest_orphan.is_some_and(|slot| slot >= end))
            && self.needs_discovery(blockstore, pinnable_slice, anchor, now)
    }

    fn probe_slots<'db>(
        &mut self,
        blockstore: &'db Blockstore,
        pinnable_slice: &mut DBPinnableSlice<'db>,
        anchor: Slot,
        quota: usize,
        outstanding: &mut HashMap<ShredRepairType, u64>,
        repairs: &mut Vec<ShredRepairType>,
    ) {
        for _ in 0..quota {
            let slot = anchor + self.next_offset;
            self.next_offset = self.next_offset % LOOKAHEAD + 1;
            if blockstore.is_dead(slot) {
                continue;
            }
            match blockstore.meta_repair_into(slot, pinnable_slice) {
                Ok(Some(meta)) if meta.received != 0 => continue,
                Err(_) => continue,
                _ => (),
            }
            // Responses supply real parent links; slot numbers do not imply ancestry.
            if let Some(repair) = RepairService::request_repair_if_needed(
                outstanding,
                ShredRepairType::HighestShred(slot, 0),
            ) {
                repairs.push(repair);
            }
        }
    }
}

#[derive(Default)]
pub(super) struct ForwardRepair {
    last_check: Option<Instant>,
    last_frontier: Option<Slot>,
    last_ancestor: Option<Slot>,
    frontiers: BTreeMap<Slot, Frontier>,
}

impl ForwardRepair {
    // Prioritize tips, but retain frozen parents: a completed child on one
    // fork does not rule out an unknown useful sibling on another fork.
    pub(super) fn frozen_frontiers(bank_forks: &BankForks) -> (BTreeSet<Slot>, BTreeSet<Slot>) {
        let root = bank_forks.root();
        let mut tips = BTreeSet::new();
        let mut parents = BTreeSet::new();
        for (slot, bank) in bank_forks.frozen_banks() {
            if slot != root {
                parents.insert(bank.parent_slot());
            }
            if slot >= root {
                tips.insert(slot);
            }
        }
        // Only retain parents that are themselves frozen candidates.
        parents.retain(|slot| tips.contains(slot));
        tips.retain(|slot| !parents.contains(slot));
        (tips, parents)
    }

    fn rotated(
        anchors: &BTreeSet<Slot>,
        after: Option<Slot>,
        limit: usize,
    ) -> impl Iterator<Item = Slot> + '_ {
        let start = after
            .and_then(|last| anchors.range((Excluded(last), Unbounded)).next())
            .or_else(|| anchors.first())
            .copied();
        start
            .into_iter()
            .flat_map(|start| anchors.range(start..).chain(anchors.range(..start)))
            .take(limit)
            .copied()
    }

    // Rate-limit checks and the total request budget across all forks.
    pub(super) fn check_due(&mut self, now: Instant) -> bool {
        if self
            .last_check
            .is_some_and(|last| now.duration_since(last) < CHECK_INTERVAL)
        {
            return false;
        }
        self.last_check = Some(now);
        true
    }

    pub(super) fn generate<'db>(
        &mut self,
        blockstore: &'db Blockstore,
        pinnable_slice: &mut DBPinnableSlice<'db>,
        anchors: &BTreeSet<Slot>,
        ancestors: &BTreeSet<Slot>,
        highest_orphan: Option<Slot>,
        outstanding: &mut HashMap<ShredRepairType, u64>,
        now: Instant,
    ) -> Vec<ShredRepairType> {
        self.frontiers
            .retain(|slot, _| anchors.contains(slot) || ancestors.contains(slot));
        for anchor in anchors.union(ancestors) {
            self.frontiers
                .entry(*anchor)
                .or_insert_with(|| Frontier::new(now));
        }
        let mut repairs = Vec::with_capacity(PROBES_PER_CHECK);
        let mut budget = PROBES_PER_CHECK;
        // At most one position goes to a stalled frozen parent. The remaining
        // positions prioritize tips; an ineligible parent consumes no positions.
        if let Some(anchor) = Self::rotated(ancestors, self.last_ancestor, 1).next() {
            self.last_ancestor = Some(anchor);
            let frontier = self.frontiers.get_mut(&anchor).unwrap();
            if frontier.eligible(blockstore, pinnable_slice, anchor, highest_orphan, now) {
                frontier.probe_slots(
                    blockstore,
                    pinnable_slice,
                    anchor,
                    1,
                    outstanding,
                    &mut repairs,
                );
                budget -= 1;
            }
        }
        // Bound tip checks/positions and rotate fairly across competing forks.
        let selected = Self::rotated(anchors, self.last_frontier, budget);
        self.last_frontier = None;
        let quota = budget / anchors.len().min(budget).max(1);
        for anchor in selected {
            self.last_frontier = Some(anchor);
            let frontier = self.frontiers.get_mut(&anchor).unwrap();
            if frontier.eligible(blockstore, pinnable_slice, anchor, highest_orphan, now) {
                frontier.probe_slots(
                    blockstore,
                    pinnable_slice,
                    anchor,
                    quota,
                    outstanding,
                    &mut repairs,
                );
            }
        }
        repairs
    }
}

#[cfg(test)]
mod tests {
    use {
        super::*,
        crate::repair::{
            repair_handler::RepairHandler, serve_repair::MAX_ORPHAN_REPAIR_RESPONSES,
            standard_repair_handler::StandardRepairHandler,
        },
        solana_ledger::{
            blockstore::make_slot_entries,
            get_tmp_ledger_path_auto_delete,
            shred::{self, Shred},
        },
        solana_perf::packet::{PacketBatch, PacketFlags},
        std::sync::Arc,
    };

    impl ForwardRepair {
        fn probe(
            &mut self,
            store: &Blockstore,
            anchor: Slot,
            highest: Option<Slot>,
            outstanding: &mut HashMap<ShredRepairType, u64>,
        ) -> Vec<ShredRepairType> {
            self.generate(
                store,
                &mut store.new_pinnable_slice(),
                &BTreeSet::from([anchor]),
                &BTreeSet::new(),
                highest,
                outstanding,
                Instant::now(),
            )
        }
    }

    fn receive(blockstore: &Blockstore, mut packets: PacketBatch) -> Vec<Slot> {
        let shreds: Vec<_> = packets
            .iter_mut()
            .map(|mut packet| {
                packet.meta_mut().flags |= PacketFlags::REPAIR;
                let (bytes, nonce) =
                    shred::layout::get_shred_and_repair_nonce(packet.as_ref()).unwrap();
                assert_eq!(nonce, Some(42));
                Shred::new_from_serialized_shred(bytes.to_vec()).unwrap()
            })
            .collect();
        let slots = shreds.iter().map(Shred::slot).collect();
        blockstore.insert_shreds(shreds, false).unwrap();
        slots
    }

    #[test]
    fn test_forward_repair_dead_and_stalled_children() {
        let path = get_tmp_ledger_path_auto_delete!();
        let store = Blockstore::open(path.path()).unwrap();
        let mut pinnable_slice = store.new_pinnable_slice();
        let (shreds, _) = make_slot_entries(101, 100, 1);
        store
            .insert_shreds(vec![shreds.last().unwrap().clone()], false)
            .unwrap();
        let anchors = BTreeSet::from([100]);
        let now = Instant::now();
        let mut probes = ForwardRepair::default();
        let mut outstanding = HashMap::new();
        assert!(
            probes
                .generate(
                    &store,
                    &mut pinnable_slice,
                    &anchors,
                    &BTreeSet::new(),
                    Some(200),
                    &mut outstanding,
                    now
                )
                .is_empty()
        );
        assert!(
            probes
                .generate(
                    &store,
                    &mut pinnable_slice,
                    &anchors,
                    &BTreeSet::new(),
                    Some(200),
                    &mut outstanding,
                    now + CHECK_INTERVAL
                )
                .is_empty()
        );
        // Received/consumed progress postpones sibling probing.
        store.insert_shreds(vec![shreds[0].clone()], false).unwrap();
        let progress_at = now + CHILD_STALL_INTERVAL;
        assert!(
            probes
                .generate(
                    &store,
                    &mut pinnable_slice,
                    &anchors,
                    &BTreeSet::new(),
                    Some(200),
                    &mut outstanding,
                    progress_at
                )
                .is_empty()
        );
        assert!(
            probes
                .generate(
                    &store,
                    &mut pinnable_slice,
                    &anchors,
                    &BTreeSet::new(),
                    Some(200),
                    &mut outstanding,
                    progress_at + CHECK_INTERVAL
                )
                .is_empty()
        );
        let repairs = probes.generate(
            &store,
            &mut pinnable_slice,
            &anchors,
            &BTreeSet::new(),
            Some(200),
            &mut outstanding,
            progress_at + CHILD_STALL_INTERVAL,
        );
        assert!(repairs.contains(&ShredRepairType::HighestShred(104, 0)));

        // A dead child must not impose even the stall grace period.
        store.set_dead_slot(101).unwrap();
        let mut probes = ForwardRepair::default();
        let repairs = probes.generate(
            &store,
            &mut pinnable_slice,
            &anchors,
            &BTreeSet::new(),
            Some(200),
            &mut HashMap::new(),
            now,
        );
        assert!(repairs.contains(&ShredRepairType::HighestShred(104, 0)));
        assert!(!repairs.iter().any(|repair| repair.slot() == 101));
        store.set_dead_slot(100).unwrap();
        assert!(
            probes
                .generate(
                    &store,
                    &mut pinnable_slice,
                    &anchors,
                    &BTreeSet::new(),
                    Some(200),
                    &mut HashMap::new(),
                    now
                )
                .is_empty()
        );
    }

    #[test]
    fn test_forward_repair_shared_budget_and_fairness() {
        let path = get_tmp_ledger_path_auto_delete!();
        let store = Blockstore::open(path.path()).unwrap();
        let mut pinnable_slice = store.new_pinnable_slice();
        let anchors: BTreeSet<_> = (1..=10).map(|i| i * 100).collect();
        let mut probes = ForwardRepair::default();
        let mut outstanding = HashMap::new();
        let now = Instant::now();
        let first = probes.generate(
            &store,
            &mut pinnable_slice,
            &anchors,
            &BTreeSet::new(),
            Some(2000),
            &mut outstanding,
            now,
        );
        assert_eq!(first.len(), PROBES_PER_CHECK);
        assert!(first.contains(&ShredRepairType::HighestShred(101, 0)));
        let second = probes.generate(
            &store,
            &mut pinnable_slice,
            &anchors,
            &BTreeSet::new(),
            Some(2000),
            &mut outstanding,
            now + CHECK_INTERVAL,
        );
        assert_eq!(second.len(), PROBES_PER_CHECK);
        assert!(second.contains(&ShredRepairType::HighestShred(901, 0)));
        assert!(second.contains(&ShredRepairType::HighestShred(1001, 0)));
        // Removed forks must not retain a cursor or stall history indefinitely.
        probes.generate(
            &store,
            &mut pinnable_slice,
            &BTreeSet::from([1000]),
            &BTreeSet::new(),
            Some(2000),
            &mut outstanding,
            now + CHECK_INTERVAL * 2,
        );
        assert_eq!(probes.frontiers.len(), 1);
    }

    #[test]
    fn test_forward_repair_gating_and_rate_limit() {
        let path = get_tmp_ledger_path_auto_delete!();
        let store = Blockstore::open(path.path()).unwrap();
        let mut probes = ForwardRepair::default();
        let mut outstanding = HashMap::new();
        let now = Instant::now();
        assert!(probes.check_due(now));
        assert!(probes.probe(&store, 100, None, &mut outstanding).is_empty());
        assert!(
            probes
                .probe(&store, 100, Some(131), &mut outstanding)
                .is_empty()
        );
        assert_eq!(
            probes.probe(&store, 100, Some(132), &mut outstanding),
            (101..=108)
                .map(|slot| ShredRepairType::HighestShred(slot, 0))
                .collect::<Vec<_>>()
        );
        assert!(!probes.check_due(now + CHECK_INTERVAL / 2));
        assert!(probes.check_due(now + CHECK_INTERVAL));
        // A different frozen bank resets the search, but not the check clock.
        assert_eq!(
            probes.probe(&store, 200, Some(300), &mut outstanding)[0],
            ShredRepairType::HighestShred(201, 0)
        );
        assert!(!probes.check_due(now + CHECK_INTERVAL));
        assert!(
            probes
                .probe(&store, Slot::MAX, Some(Slot::MAX), &mut outstanding)
                .is_empty()
        );
        // Once a successor is known, ordinary repair handles its missing data.
        let (shreds, _) = make_slot_entries(204, 200, 1);
        store
            .insert_shreds(vec![shreds.last().unwrap().clone()], false)
            .unwrap();
        assert!(
            probes
                .probe(&store, 200, Some(300), &mut outstanding)
                .is_empty()
        );
    }

    #[test]
    fn test_forward_repair_rotates_retries_and_deduplicates() {
        let path = get_tmp_ledger_path_auto_delete!();
        let store = Blockstore::open(path.path()).unwrap();
        let mut probes = ForwardRepair::default();
        let mut outstanding = HashMap::new();
        // This slot belongs to another parent; don't download it again and
        // don't mistake its numeric proximity for a link to the frozen bank.
        let (shreds, _) = make_slot_entries(102, 99, 1);
        store.insert_shreds(shreds, false).unwrap();
        outstanding.insert(ShredRepairType::HighestShred(103, 0), 0);
        let mut requested = Vec::new();
        for _ in 0..4 {
            requested.extend(probes.probe(&store, 100, Some(300), &mut outstanding));
        }
        assert_eq!(
            requested,
            (101..=132)
                .filter(|slot| ![102, 103].contains(slot))
                .map(|slot| ShredRepairType::HighestShred(slot, 0))
                .collect::<Vec<_>>()
        );
        assert!(
            probes
                .probe(&store, 100, Some(300), &mut outstanding)
                .is_empty()
        );
        // Simulate expiration by the normal repair loop: lost requests retry
        // on the next pass instead of extending the speculative search range.
        outstanding.clear();
        let retried = probes.probe(&store, 100, Some(300), &mut outstanding);
        assert_eq!(
            retried,
            (109..=116)
                .map(|slot| ShredRepairType::HighestShred(slot, 0))
                .collect::<Vec<_>>()
        );
        // A switch to a lower frozen fork also resets the cursor.
        assert_eq!(
            probes.probe(&store, 50, Some(300), &mut outstanding)[0],
            ShredRepairType::HighestShred(51, 0)
        );
    }

    // Deterministic network-round simulation with production request handlers
    // and blockstore insertion. No sockets, sleeps, or validator are needed.
    fn simulate_discovery(skipped: Slot) {
        const BASE: Slot = 100;
        const TIP: Slot = 228;
        let remote_path = get_tmp_ledger_path_auto_delete!();
        let baseline_path = get_tmp_ledger_path_auto_delete!();
        let forward_path = get_tmp_ledger_path_auto_delete!();
        let remote = Arc::new(Blockstore::open(remote_path.path()).unwrap());
        let baseline = Blockstore::open(baseline_path.path()).unwrap();
        let forward = Blockstore::open(forward_path.path()).unwrap();
        let first_child = BASE + skipped + 1;
        let mut parent = BASE - 1;
        for slot in std::iter::once(BASE).chain(first_child..=TIP) {
            let (shreds, _) = make_slot_entries(slot, parent, 1);
            if slot == BASE {
                baseline.insert_shreds(shreds.clone(), false).unwrap();
                forward.insert_shreds(shreds.clone(), false).unwrap();
            } else if slot == TIP {
                baseline
                    .insert_shreds(vec![shreds.last().unwrap().clone()], false)
                    .unwrap();
                forward
                    .insert_shreds(vec![shreds.last().unwrap().clone()], false)
                    .unwrap();
            }
            remote.insert_shreds(shreds, false).unwrap();
            parent = slot;
        }
        let handler = StandardRepairHandler::new(remote);
        let addr = "127.0.0.1:1234".parse().unwrap();
        let mut cursor = TIP - 1;
        let mut baseline_rounds = 0;
        while baseline.meta(BASE).unwrap().unwrap().next_slots.is_empty() {
            let response = handler
                .run_orphan(&addr, cursor, MAX_ORPHAN_REPAIR_RESPONSES, 42)
                .unwrap();
            let slots = receive(&baseline, response);
            cursor = baseline
                .meta(*slots.iter().min().unwrap())
                .unwrap()
                .unwrap()
                .parent_slot
                .unwrap();
            baseline_rounds += 1;
            assert!(baseline_rounds <= 13);
        }

        let mut probes = ForwardRepair::default();
        let mut outstanding = HashMap::new();
        let start = Instant::now();
        let mut forward_rounds = 0;
        while forward.meta(BASE).unwrap().unwrap().next_slots.is_empty() {
            assert!(probes.check_due(start + CHECK_INTERVAL * forward_rounds));
            for request in probes.probe(&forward, BASE, Some(TIP - 1), &mut outstanding) {
                let ShredRepairType::HighestShred(slot, index) = request else {
                    panic!("unexpected request")
                };
                if let Some(response) = handler.run_highest_window_request(&addr, slot, index, 42) {
                    receive(&forward, response);
                }
            }
            forward_rounds += 1;
            assert!(forward_rounds <= 4);
        }
        assert_eq!(
            forward.meta(BASE).unwrap().unwrap().next_slots.as_slice(),
            &[first_child]
        );
        assert_eq!(forward_rounds as u64, skipped / PROBES_PER_CHECK as u64 + 1);
        assert!(forward_rounds < baseline_rounds);

        // This test measures discovery only. Production selection and replay
        // of the discovered child are covered by repair_service tests.
        assert!(!forward.meta(first_child).unwrap().unwrap().is_full());
        assert!(
            probes
                .probe(&forward, BASE, Some(TIP), &mut outstanding)
                .is_empty()
        );
        eprintln!(
            "skipped={skipped}: backward discovery={baseline_rounds} rounds, forward \
             discovery={forward_rounds} rounds"
        );
    }

    #[test]
    fn test_forward_repair_long_gap() {
        simulate_discovery(0);
    }

    #[test]
    fn test_forward_repair_skipped_slots() {
        simulate_discovery(3);
        simulate_discovery(10);
    }
}

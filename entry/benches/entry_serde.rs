use {
    criterion::{criterion_group, criterion_main, Criterion},
    rand::prelude::*,
    solana_address::Address,
    solana_entry::entry,
    solana_hash::Hash,
    solana_message::{
        v0::{self, MessageAddressTableLookup},
        MessageHeader, VersionedMessage,
    },
    solana_signature::Signature,
    solana_transaction::{versioned::VersionedTransaction, CompiledInstruction},
    std::{array, hint::black_box},
};

fn rand_message_header() -> MessageHeader {
    let mut rng = rand::thread_rng();
    MessageHeader {
        num_required_signatures: rng.gen(),
        num_readonly_signed_accounts: rng.gen(),
        num_readonly_unsigned_accounts: rng.gen(),
    }
}

fn rand_account_keys() -> Vec<Address> {
    (0..32).map(|_| Address::new_unique()).collect()
}

fn rand_signatures() -> Vec<Signature> {
    (0..32)
        .map(|_| Signature::from(array::from_fn(|_| rand::thread_rng().gen())))
        .collect()
}

fn rand_byte_vec() -> Vec<u8> {
    (0..1024).map(|_| rand::thread_rng().gen()).collect()
}

fn rand_instructions() -> Vec<CompiledInstruction> {
    let mut rng = rand::thread_rng();
    (0..8)
        .map(|_| {
            CompiledInstruction::new_from_raw_parts(rng.gen(), rand_byte_vec(), rand_byte_vec())
        })
        .collect()
}

fn rand_address_table_lookups() -> Vec<MessageAddressTableLookup> {
    (0..8)
        .map(|_| MessageAddressTableLookup {
            account_key: Address::new_unique(),
            writable_indexes: rand_byte_vec(),
            readonly_indexes: rand_byte_vec(),
        })
        .collect()
}

fn rand_v0_message() -> v0::Message {
    v0::Message {
        header: rand_message_header(),
        account_keys: rand_account_keys(),
        recent_blockhash: Hash::new_unique(),
        instructions: rand_instructions(),
        address_table_lookups: rand_address_table_lookups(),
    }
}

fn rand_versioned_message() -> VersionedMessage {
    VersionedMessage::V0(rand_v0_message())
}

fn rand_versioned_transaction() -> VersionedTransaction {
    VersionedTransaction {
        signatures: rand_signatures(),
        message: rand_versioned_message(),
    }
}

fn rand_transactions() -> Vec<VersionedTransaction> {
    (0..8).map(|_| rand_versioned_transaction()).collect()
}

fn rand_entry() -> entry::Entry {
    let mut rng = rand::thread_rng();
    let num_hashes = rng.gen();
    let hash = Hash::new_unique();
    let transactions = rand_transactions();
    entry::Entry {
        num_hashes,
        hash,
        transactions,
    }
}

fn rand_entries() -> Vec<entry::Entry> {
    (0..32).map(|_| rand_entry()).collect()
}

fn bench_ser(c: &mut Criterion) {
    let mut group = c.benchmark_group("ser");
    let entry = rand_entry();
    let entry_inner = entry.clone();
    let entries_inner = rand_entries();

    group.bench_function("bincode_ser_entry", |b| {
        b.iter(|| {
            let bytes = bincode::serialize(black_box(&entry_inner)).unwrap();
            black_box(&bytes);
        });
    });
    group.bench_function("bincode_ser_entries", |b| {
        b.iter(|| {
            let bytes = bincode::serialize(black_box(&entries_inner)).unwrap();
            black_box(&bytes);
        });
    });
    group.bench_function("wincode_ser_entry", |b| {
        b.iter(|| {
            let bytes = wincode::serialize(black_box(&entry_inner)).unwrap();
            black_box(&bytes);
        });
    });
    group.bench_function("wincode_ser_entries", |b| {
        b.iter(|| {
            let bytes = wincode::serialize(black_box(&entries_inner)).unwrap();
            black_box(&bytes);
        });
    });
}

fn bench_ser_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("ser_size");
    let entry = rand_entry();
    let entries = rand_entries();

    group.bench_function("bincode_ser_size_entry", |b| {
        b.iter(|| {
            let size = bincode::serialized_size(black_box(&entry)).unwrap();
            black_box(&size);
        });
    });
    group.bench_function("bincode_ser_size_entries", |b| {
        b.iter(|| {
            let size = bincode::serialized_size(black_box(&entries)).unwrap();
            black_box(&size);
        });
    });
    group.bench_function("wincode_ser_size_entry", |b| {
        b.iter(|| {
            let size = wincode::serialized_size(black_box(&entry)).unwrap();
            black_box(&size);
        });
    });
    group.bench_function("wincode_ser_size_entries", |b| {
        b.iter(|| {
            let size = wincode::serialized_size(black_box(&entries)).unwrap();
            black_box(&size);
        });
    });
}

fn bench_deser(c: &mut Criterion) {
    let mut group = c.benchmark_group("deser");
    let entry = rand_entry();
    let entry_serialized = bincode::serialize(&entry).unwrap();
    let entries = rand_entries();
    let entries_serialized = bincode::serialize(&entries).unwrap();
    let num_entries = entries.len();
    println!("num entries: {}", num_entries);
    println!("entries size: {}", entries_serialized.len());
    println!("avg entry size: {}", entries_serialized.len() / num_entries);

    group.bench_function("bincode_deser_entry", |b| {
        b.iter(|| {
            let entry_val: entry::Entry =
                bincode::deserialize(black_box(&entry_serialized)).unwrap();
            black_box(&entry_val);
        });
    });
    group.bench_function("bincode_deser_entries", |b| {
        b.iter(|| {
            let entries_val: Vec<entry::Entry> =
                bincode::deserialize(black_box(&entries_serialized)).unwrap();
            black_box(&entries_val);
        });
    });
    group.bench_function("wincode_deser_entry", |b| {
        b.iter(|| {
            let entry_val: entry::Entry =
                wincode::deserialize(black_box(&entry_serialized)).unwrap();
            black_box(&entry_val);
        });
    });
    group.bench_function("wincode_deser_entries", |b| {
        b.iter(|| {
            let entries_val: Vec<entry::Entry> =
                wincode::deserialize(black_box(&entries_serialized)).unwrap();
            black_box(&entries_val);
        });
    });
}

criterion_group!(benches, bench_deser, bench_ser, bench_ser_size);
criterion_main!(benches);

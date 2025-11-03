use std::{
    cell::Cell,
    iter::{Enumerate, FusedIterator},
    marker::PhantomData,
    mem,
    ops::{Index, RangeInclusive},
};

/// Trait for converting between a key and a usize.
///
/// Only implemented for types that can be converted infallibly to usize and back.
///
/// We use this to avoid the noise and unwraps that would be required by using
/// `TryInto<usize>` and `TryFrom<usize>` in the implementation.
pub trait AsKey: Copy {
    fn into_key(self) -> usize;
    fn from_key(key: usize) -> Self;
}

macro_rules! impl_as_key {
    ($ty:ty) => {
        impl AsKey for $ty {
            #[inline(always)]
            fn into_key(self) -> usize {
                self as usize
            }

            #[inline(always)]
            fn from_key(key: usize) -> Self {
                key as $ty
            }
        }
    };
}

impl_as_key!(usize);
#[cfg(target_pointer_width = "64")]
impl_as_key!(u64);
impl_as_key!(u32);
impl_as_key!(u16);
impl_as_key!(u8);

/// Flat index map backed by a single contiguous allocation.
///
/// `IndexMap` is optimal for random access on dense set of integer keys.
/// Keys are extremely cheap to compute, as they're simply offsets relative
/// to the minimum key in the map (i.e. the start of the range).
/// Access is similarly cheap when ranges are known ahead of time and space
/// is pre-allocated, as keys map directly to the underlying buffer
/// -- O(1) lookup without node traversal / pointer chasing.
///
/// While the implementation gracefully handles growing, it is *highly* recommended
/// to use this on a dense range of keys that is known ahead of time. Because the underlying
/// buffer is a single contiguous allocation, space requirement is max(key) - min(key) + 1.
/// Growing requires allocating additional space to accommodate the delta between the current
/// edge of the buffer and the new key; no allocation is needed if the new key is within the current
/// range. For example:
/// ```ignore
/// // Initial allocation is 11 elements.
/// let range = 0..=10;
/// // Insert at key 5 (within range), no allocation is needed.
/// insert(5, value);
/// // To insert at key 100, we need to allocate space for 90 more elements.
/// insert(100, value);
/// ```
#[derive(Default, Debug, Clone)]
pub struct IndexMap<K, V> {
    inner: Vec<Option<V>>,
    start: usize,
    end: usize,
    len: usize,
    _k: PhantomData<K>,
}

impl<K, V> IndexMap<K, V> {
    pub fn new() -> Self {
        Self {
            inner: Vec::new(),
            start: 0,
            end: 0,
            len: 0,
            _k: PhantomData,
        }
    }

    /// Clears the map, resetting it to the initial state, preserving the capacity.
    pub fn clear(&mut self) {
        self.inner.clear();
        self.start = 0;
        self.end = 0;
        self.len = 0;
    }

    /// Ensures that the map has capacity for the given key.
    ///
    /// - If the map is empty, sets the start and end to the given key.
    /// - If the key is greater than the current end, extends the buffer to the right.
    /// - If the key is less than the current start, extends the buffer to the left.
    #[inline]
    fn ensure_capacity(&mut self, key: usize) {
        if self.inner.is_empty() {
            self.inner.push(None);
            self.start = key;
            self.end = key;
            return;
        }
        if key > self.end {
            self.inner.extend((0..(key - self.end)).map(|_| None));
            self.end = key;
            return;
        }
        if key < self.start {
            let n = self.start - key;
            self.inner.extend((0..n).map(|_| None));
            self.inner.rotate_right(n);
            self.start = key;
        }
    }

    fn ensure_capacity_in(&mut self, range: RangeInclusive<usize>) {
        let start = *range.start();
        let end = *range.end();
        if self.inner.is_empty() {
            self.inner.extend((start..=end).map(|_| None));
            self.start = start;
            self.end = end;
            return;
        }
        if end > self.end {
            self.inner.extend((0..(end - self.end)).map(|_| None));
            self.end = end;
        }
        if start < self.start {
            let n = self.start - start;
            self.inner.extend((0..n).map(|_| None));
            self.inner.rotate_right(n);
            self.start = start;
        }
    }

    /// Checks if the map contains the given key.
    #[inline]
    fn contains_key_inner(&self, key: usize) -> bool {
        key >= self.start && key <= self.end && !self.inner.is_empty()
    }

    /// Maps a key to the underlying buffer index.
    ///
    /// Returns `None` if the key is out of bounds.
    #[inline]
    fn mapped_key(&self, key: usize) -> Option<usize> {
        if !self.contains_key_inner(key) {
            return None;
        }
        // SAFETY: `contains_key_inner` ensures the key is in bounds.
        Some(self.mapped_key_unchecked(key))
    }

    /// Maps a key to the underlying buffer index without
    /// checking if the key is in bounds.
    #[inline]
    fn mapped_key_unchecked(&self, key: usize) -> usize {
        key - self.start
    }

    /// Returns the number of active elements in the map.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns whether the map has no active elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns an iterator over the map's active key-value pairs by reference in
    /// ascending key order.
    pub fn iter(&self) -> IndexMapIter<'_, K, V> {
        IndexMapIter {
            inner: self.inner.iter().enumerate(),
            start: self.start,
            _k: PhantomData,
            remaining: self.len,
        }
    }
}

impl<K, V> IndexMap<K, V>
where
    K: AsKey,
{
    /// Creates a new `IndexMap` with capacity for the given range.
    ///
    /// The map will be pre-allocated to hold at least the number of elements
    /// in the range.
    pub fn with_capacity_for_range(range: RangeInclusive<K>) -> Self {
        let start = range.start().into_key();
        let end = range.end().into_key().max(start);
        Self {
            inner: (start..=end).map(|_| None).collect(),
            start,
            end,
            len: 0,
            _k: PhantomData,
        }
    }

    /// Reserves capacity for the given range.
    ///
    /// This function does not allocate if the range is already covered by the current capacity.
    pub fn reserve(&mut self, range: RangeInclusive<K>) {
        let start = range.start().into_key();
        let end = range.end().into_key().max(start);
        self.ensure_capacity_in(start..=end);
    }

    /// Returns a mutable entry for the given key.
    ///
    /// If the key is already present, returns [`IndexMapEntry::Occupied`].
    /// If the key is not present, returns [`IndexMapEntry::Vacant`].
    pub fn entry(&mut self, key: K) -> IndexMapEntry<'_, K, V> {
        let k = key.into_key();
        self.ensure_capacity(k);
        // SAFETY: `ensure_capacity` ensures the key is in bounds.
        let k = self.mapped_key_unchecked(k);
        let slot = &mut self.inner[k];
        if slot.is_some() {
            return IndexMapEntry::Occupied(OccupiedEntry {
                key,
                value: slot,
                len: Cell::from_mut(&mut self.len),
            });
        }

        IndexMapEntry::Vacant(VacantEntry {
            key,
            value: slot,
            len: Cell::from_mut(&mut self.len),
        })
    }

    pub fn entries(&mut self, keys: RangeInclusive<K>) -> EntriesIter<'_, K, V> {
        let min = keys.start().into_key();
        let max = keys.end().into_key();

        if min > max {
            return EntriesIter {
                inner: [].iter_mut().enumerate(),
                start: min,
                len: Cell::from_mut(&mut self.len),
                _k: PhantomData,
            };
        }

        self.ensure_capacity_in(min..=max);
        let lo = self.mapped_key_unchecked(min);
        let hi = self.mapped_key_unchecked(max);
        EntriesIter {
            inner: self.inner[lo..=hi].iter_mut().enumerate(),
            start: min,
            len: Cell::from_mut(&mut self.len),
            _k: PhantomData,
        }
    }

    pub fn entries_from_iter<'a>(
        &mut self,
        keys: impl IntoIterator<Item = &'a K>,
    ) -> EntriesIter<'_, K, V>
    where
        K: 'a,
    {
        let mut min = usize::MAX;
        let mut max = 0;
        for key in keys {
            min = min.min(key.into_key());
            max = max.max(key.into_key());
        }
        self.entries(K::from_key(min)..=K::from_key(max))
    }

    /// Inserts a key-value pair into the map.
    ///
    /// If the key is already present, the value is replaced.
    pub fn insert(&mut self, key: K, value: V) {
        let key = key.into_key();
        self.ensure_capacity(key);
        // SAFETY: `ensure_capacity` ensures the key is in bounds.
        let key = self.mapped_key_unchecked(key);
        let slot = &mut self.inner[key];
        if slot.is_none() {
            self.len += 1;
        }
        *slot = Some(value);
    }

    /// Returns a reference to the value corresponding to the key.
    ///
    /// Returns `None` if the key does not contain a value.
    #[inline]
    pub fn get(&self, key: K) -> Option<&V> {
        let key = self.mapped_key(key.into_key())?;
        self.inner[key].as_ref()
    }

    /// Checks if the map contains the given key.
    #[inline]
    pub fn contains_key(&self, key: K) -> bool {
        self.contains_key_inner(key.into_key())
    }
}

/// A view into an occupied entry in an [`IndexMap`].
/// A variant of the [`IndexMapEntry`] enum.
pub struct OccupiedEntry<'a, K, V> {
    key: K,
    value: &'a mut Option<V>,
    len: &'a Cell<usize>,
}

impl<'a, K, V> OccupiedEntry<'a, K, V> {
    /// Gets a reference to the value in the entry.
    #[inline]
    pub fn get(&self) -> &V {
        // SAFETY: `OccupiedEntry` is only constructed if the value is `Some`.
        unsafe { self.value.as_ref().unwrap_unchecked() }
    }

    /// Gets a reference to the key in the entry.
    #[inline]
    pub fn key(&self) -> &K {
        &self.key
    }

    /// Gets a mutable reference to the value in the entry.
    ///
    /// If you need a reference to the `OccupiedEntry` which may outlive the
    /// destruction of the `Entry` value, see [`OccupiedEntry::into_mut`].
    #[inline]
    pub fn get_mut(&mut self) -> &mut V {
        // SAFETY: `OccupiedEntry` is only constructed if the value is `Some`.
        unsafe { self.value.as_mut().unwrap_unchecked() }
    }

    /// Converts the `OccupiedEntry` into a mutable reference to the value in the
    /// entry with a lifetime bound to the map itself.
    ///
    /// If you need multiple references to the `OccupiedEntry`, use [`OccupiedEntry::get_mut`].
    #[inline]
    pub fn into_mut(self) -> &'a mut V {
        // SAFETY: `OccupiedEntry` is only constructed if the value is `Some`.
        unsafe { self.value.as_mut().unwrap_unchecked() }
    }

    /// Sets the value of the entry and returns the entry's old value.
    #[inline]
    pub fn insert(&mut self, value: V) -> V {
        let v = mem::replace(self.value, Some(value));
        // SAFETY: `OccupiedEntry` is only constructed if the value is `Some`.
        unsafe { v.unwrap_unchecked() }
    }

    /// Takes the value out of the entry, and returns it.
    pub fn remove(self) -> V {
        self.len.set(self.len.get() - 1);
        let v = mem::take(self.value);
        // SAFETY: `OccupiedEntry` is only constructed if the value is `Some`.
        unsafe { v.unwrap_unchecked() }
    }
}

/// A view into a vacant entry in an [`IndexMap`].
/// A variant of the [`IndexMapEntry`] enum.
pub struct VacantEntry<'a, K, V> {
    key: K,
    value: &'a mut Option<V>,
    len: &'a Cell<usize>,
}

impl<'a, K, V> VacantEntry<'a, K, V> {
    /// Gets a reference to the key that would be used when inserting a value through the `VacantEntry`.
    #[inline]
    pub fn key(&self) -> &K {
        &self.key
    }

    /// Sets the value of the entry with the `VacantEntry`'s key,
    /// and returns a mutable reference to it.
    #[inline]
    pub fn insert(self, value: V) -> &'a mut V {
        *self.value = Some(value);
        self.len.set(self.len.get() + 1);
        // SAFETY: We just set the value to `Some`, so it must be `Some`.
        unsafe { self.value.as_mut().unwrap_unchecked() }
    }
}
pub enum IndexMapEntry<'a, K, V> {
    Occupied(OccupiedEntry<'a, K, V>),
    Vacant(VacantEntry<'a, K, V>),
}

impl<'a, K, V: Default> IndexMapEntry<'a, K, V> {
    /// Ensures a value is in the entry by inserting the default value if empty,
    /// and returns a mutable reference to the value in the entry.
    #[inline]
    pub fn or_default(self) -> &'a mut V {
        match self {
            IndexMapEntry::Occupied(entry) => entry.into_mut(),
            IndexMapEntry::Vacant(entry) => entry.insert(Default::default()),
        }
    }
}

impl<K, V> Index<K> for IndexMap<K, V>
where
    K: AsKey,
{
    type Output = V;

    #[inline]
    fn index(&self, index: K) -> &Self::Output {
        self.inner[self.mapped_key(index.into_key()).unwrap()]
            .as_ref()
            .unwrap()
    }
}

impl<K, V> FromIterator<(K, V)> for IndexMap<K, V>
where
    K: AsKey,
{
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        let mut slf = Self::new();
        for (key, value) in iter {
            slf.insert(key, value);
        }
        slf
    }
}

#[derive(Debug)]
pub struct EntriesIter<'a, K, V> {
    inner: Enumerate<std::slice::IterMut<'a, Option<V>>>,
    start: usize,
    len: &'a Cell<usize>,
    _k: PhantomData<K>,
}

impl<'a, K, V> Iterator for EntriesIter<'a, K, V>
where
    K: AsKey,
{
    type Item = IndexMapEntry<'a, K, V>;

    fn next(&mut self) -> Option<Self::Item> {
        let (i, value) = self.inner.next()?;
        let key = K::from_key(i + self.start);
        if value.is_some() {
            return Some(IndexMapEntry::Occupied(OccupiedEntry {
                key,
                value,
                len: self.len,
            }));
        }
        Some(IndexMapEntry::Vacant(VacantEntry {
            key,
            value,
            len: self.len,
        }))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<K, V> ExactSizeIterator for EntriesIter<'_, K, V>
where
    K: AsKey,
{
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<K, V> FusedIterator for EntriesIter<'_, K, V> where K: AsKey {}

impl<K, V> Index<&K> for IndexMap<K, V>
where
    K: AsKey,
{
    type Output = V;

    #[inline]
    fn index(&self, index: &K) -> &Self::Output {
        &self[*index]
    }
}

/// Recovers a key from its relative index and start.
#[inline]
fn recover_key<K>(index: usize, start: usize) -> K
where
    K: AsKey,
{
    K::from_key(index + start)
}

#[derive(Debug, Clone)]
pub struct IndexMapIter<'a, K, V> {
    inner: Enumerate<std::slice::Iter<'a, Option<V>>>,
    start: usize,
    _k: PhantomData<K>,
    remaining: usize,
}

impl<'a, K, V> Iterator for IndexMapIter<'a, K, V>
where
    K: AsKey,
{
    type Item = (K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        for (i, v) in self.inner.by_ref() {
            if let Some(v) = v {
                self.remaining -= 1;
                return Some((recover_key(i, self.start), v));
            }
        }
        None
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<K, V> FusedIterator for IndexMapIter<'_, K, V> where K: AsKey {}

impl<K, V> ExactSizeIterator for IndexMapIter<'_, K, V>
where
    K: AsKey,
{
    #[inline]
    fn len(&self) -> usize {
        self.remaining
    }
}

impl<'a, K, V> IntoIterator for &'a IndexMap<K, V>
where
    K: AsKey,
{
    type Item = (K, &'a V);
    type IntoIter = IndexMapIter<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        IndexMapIter {
            inner: self.inner.iter().enumerate(),
            start: self.start,
            _k: PhantomData,
            remaining: self.len,
        }
    }
}

#[derive(Debug, Clone)]
pub struct IndexMapIntoIter<K, V> {
    inner: Enumerate<std::vec::IntoIter<Option<V>>>,
    start: usize,
    remaining: usize,
    _k: PhantomData<K>,
}

impl<K, V> FusedIterator for IndexMapIntoIter<K, V> where K: AsKey {}

impl<K, V> Iterator for IndexMapIntoIter<K, V>
where
    K: AsKey,
{
    type Item = (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        for (i, v) in self.inner.by_ref() {
            if let Some(v) = v {
                self.remaining -= 1;
                return Some((recover_key(i, self.start), v));
            }
        }
        None
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<K, V> ExactSizeIterator for IndexMapIntoIter<K, V>
where
    K: AsKey,
{
    #[inline]
    fn len(&self) -> usize {
        self.remaining
    }
}

impl<K, V> IntoIterator for IndexMap<K, V>
where
    K: AsKey,
{
    type Item = (K, V);
    type IntoIter = IndexMapIntoIter<K, V>;

    fn into_iter(self) -> Self::IntoIter {
        IndexMapIntoIter {
            inner: self.inner.into_iter().enumerate(),
            start: self.start,
            remaining: self.len,
            _k: PhantomData,
        }
    }
}

#[derive(Default, Debug, Clone)]
pub struct IndexSet<K> {
    inner: IndexMap<K, ()>,
}

impl<K> IndexSet<K> {
    pub fn new() -> Self {
        Self {
            inner: IndexMap::new(),
        }
    }

    /// Clears the set, resetting it to the initial state, preserving the capacity.
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// Returns the number of active elements in the set.
    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns whether the set has no active elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns an iterator over the set's active keys by reference in
    /// ascending key order
    pub fn iter(&self) -> IndexSetIter<'_, K> {
        IndexSetIter {
            inner: self.inner.inner.iter().enumerate(),
            start: self.inner.start,
            _k: PhantomData,
            remaining: self.inner.len,
        }
    }
}

impl<K> IndexSet<K>
where
    K: AsKey,
{
    /// Creates a new `IndexSet` with capacity for the given range.
    ///
    /// The set will be pre-allocated to hold at least the number of elements
    /// in the range.
    pub fn with_capacity_for_range(range: RangeInclusive<K>) -> Self {
        Self {
            inner: IndexMap::with_capacity_for_range(range),
        }
    }

    /// Reserves capacity for the given range.
    ///
    /// This function does not allocate if the range is already covered by the current capacity.
    pub fn reserve(&mut self, range: RangeInclusive<K>) {
        self.inner.reserve(range);
    }

    /// Inserts a key into the set.
    ///
    /// No effect if the key is already present.
    #[inline]
    pub fn insert(&mut self, key: K) {
        self.inner.insert(key, ());
    }

    /// Checks if the set contains the given key.
    #[inline]
    pub fn contains(&self, key: K) -> bool {
        self.inner.contains_key(key)
    }
}

#[derive(Debug, Clone)]
pub struct IndexSetIter<'a, K> {
    inner: Enumerate<std::slice::Iter<'a, Option<()>>>,
    start: usize,
    remaining: usize,
    _k: PhantomData<K>,
}

impl<K> FusedIterator for IndexSetIter<'_, K> where K: AsKey {}

impl<K> Iterator for IndexSetIter<'_, K>
where
    K: AsKey,
{
    type Item = K;

    fn next(&mut self) -> Option<Self::Item> {
        for (i, v) in self.inner.by_ref() {
            if v.is_some() {
                self.remaining -= 1;
                return Some(recover_key(i, self.start));
            }
        }
        None
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<K> ExactSizeIterator for IndexSetIter<'_, K>
where
    K: AsKey,
{
    #[inline]
    fn len(&self) -> usize {
        self.remaining
    }
}

impl<'a, K> IntoIterator for &'a IndexSet<K>
where
    K: AsKey,
{
    type Item = K;
    type IntoIter = IndexSetIter<'a, K>;

    fn into_iter(self) -> Self::IntoIter {
        IndexSetIter {
            inner: self.inner.inner.iter().enumerate(),
            start: self.inner.start,
            _k: PhantomData,
            remaining: self.inner.len,
        }
    }
}

#[derive(Debug, Clone)]
pub struct IndexSetIntoIter<K> {
    inner: Enumerate<std::vec::IntoIter<Option<()>>>,
    start: usize,
    remaining: usize,
    _k: PhantomData<K>,
}

impl<K> FusedIterator for IndexSetIntoIter<K> where K: AsKey {}

impl<K> Iterator for IndexSetIntoIter<K>
where
    K: AsKey,
{
    type Item = K;

    fn next(&mut self) -> Option<Self::Item> {
        for (i, v) in self.inner.by_ref() {
            if v.is_some() {
                self.remaining -= 1;
                return Some(recover_key(i, self.start));
            }
        }
        None
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<K> ExactSizeIterator for IndexSetIntoIter<K>
where
    K: AsKey,
{
    #[inline]
    fn len(&self) -> usize {
        self.remaining
    }
}

impl<K> IntoIterator for IndexSet<K>
where
    K: AsKey,
{
    type Item = K;
    type IntoIter = IndexSetIntoIter<K>;

    fn into_iter(self) -> Self::IntoIter {
        IndexSetIntoIter {
            inner: self.inner.inner.into_iter().enumerate(),
            start: self.inner.start,
            _k: PhantomData,
            remaining: self.inner.len,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entry_len_and_replace_and_remove() {
        let mut m: IndexMap<u64, u64> = IndexMap::with_capacity_for_range(0u64..=2u64);
        assert_eq!(m.len(), 0);

        match m.entry(1) {
            IndexMapEntry::Vacant(v) => {
                let r = v.insert(10);
                assert_eq!(*r, 10);
            }
            _ => panic!("expected vacant"),
        }
        assert_eq!(m.len(), 1);

        match m.entry(1) {
            IndexMapEntry::Occupied(mut o) => {
                assert_eq!(*o.get(), 10);
                let old = o.insert(11);
                assert_eq!(old, 10);
                assert_eq!(*o.get(), 11);
            }
            _ => panic!("expected occupied"),
        }
        assert_eq!(m.len(), 1);

        match m.entry(1) {
            IndexMapEntry::Occupied(o) => {
                let removed = o.remove();
                assert_eq!(removed, 11);
            }
            _ => panic!("expected occupied"),
        }
        assert_eq!(m.len(), 0);
        assert!(m.get(1).is_none());
    }

    #[test]
    fn insert_get_and_index_and_bounds() {
        let mut m: IndexMap<u64, u64> = IndexMap::with_capacity_for_range(3u64..=7u64);
        assert!(m.get(3).is_none());
        m.insert(3, 1);
        m.insert(7, 2);
        assert_eq!(m.len(), 2);
        assert_eq!(m.get(3), Some(&1));
        assert_eq!(m.get(7), Some(&2));
        assert_eq!(m[3], 1);
        assert_eq!(m[7], 2);
    }

    #[test]
    #[should_panic]
    fn indexing_missing_panics() {
        let m: IndexMap<u64, u64> = IndexMap::with_capacity_for_range(0u64..=3u64);
        let _ = m[0];
    }

    #[test]
    fn iter_exact_size_and_items() {
        let mut m: IndexMap<u64, u64> = IndexMap::with_capacity_for_range(0u64..=4u64);
        m.insert(0, 10);
        m.insert(3, 30);
        assert_eq!(m.len(), 2);

        let it = m.iter();
        let (lo, hi) = it.size_hint();
        assert_eq!(lo, 2);
        assert_eq!(hi, Some(2));

        let items: Vec<_> = m.iter().collect();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0], (0u64, &10));
        assert_eq!(items[1], (3u64, &30));
    }

    #[test]
    fn into_iter_exact_size_and_progress() {
        let mut m: IndexMap<u64, u64> = IndexMap::with_capacity_for_range(10u64..=12u64);
        m.insert(10, 1);
        m.insert(12, 3);
        let mut it = m.iter();
        assert_eq!(it.size_hint(), (2, Some(2)));
        assert_eq!(it.len(), 2);
        let _ = it.next();
        assert_eq!(it.len(), 1);
        let _ = it.next();
        assert_eq!(it.len(), 0);
        assert!(it.next().is_none());
    }

    #[test]
    fn grows_correctly() {
        let mut m: IndexMap<u64, u64> = IndexMap::new();
        m.insert(1, 1);
        assert_eq!(m.len(), 1);
        // rotate
        m.insert(0, 0);
        assert_eq!(m.len(), 2);
        m.insert(10, 2);
        assert_eq!(m.len(), 3);
        m.insert(200, 3);
        assert_eq!(m.len(), 4);

        assert_eq!(
            &m.into_iter().collect::<Vec<_>>(),
            &[(0, 0), (1, 1), (10, 2), (200, 3)]
        );
    }

    #[test]
    fn test_entries() {
        let mut m: IndexMap<u64, u64> = IndexMap::new();
        m.insert(3, 3);
        m.insert(7, 7);
        m.entries(0..=5)
            .enumerate()
            .for_each(|(i, entry)| match entry {
                IndexMapEntry::Occupied(_) => {}
                IndexMapEntry::Vacant(v) => {
                    v.insert(i as u64 * 2);
                }
            });

        assert_eq!(
            m.into_iter().collect::<Vec<_>>(),
            [(0, 0), (1, 2), (2, 4), (3, 3), (4, 8), (5, 10), (7, 7)]
        );
    }

    #[test]
    fn test_entries_empty_range_is_empty_and_noop() {
        let mut m: IndexMap<u64, u64> = IndexMap::new();
        m.insert(2, 10);
        let len_before = m.len();
        #[allow(clippy::reversed_empty_ranges)]
        let mut it = m.entries(5..=4);
        assert_eq!(it.size_hint(), (0, Some(0)));
        assert!(it.next().is_none());
        assert_eq!(m.len(), len_before);
        // Ensure contents unchanged
        assert_eq!(m.into_iter().collect::<Vec<_>>(), [(2, 10)]);
    }

    #[test]
    fn test_entries_extends_both_sides_and_fills_new_only() {
        let mut m: IndexMap<u64, u64> = IndexMap::new();
        m.insert(10, 1);
        let len_before = m.len();
        // Fill 8..=12; should grow front (below 10) and back (above 10)
        m.entries(8..=12).for_each(|entry| match entry {
            IndexMapEntry::Occupied(mut o) => {
                // Overwrite occupied to verify it doesn't change len
                o.insert(99);
            }
            IndexMapEntry::Vacant(v) => {
                v.insert(42);
            }
        });
        // New keys: 8,9,11,12 → +4, occupied 10 replaced in place
        assert_eq!(m.len(), len_before + 4);
        // Verify values
        let got = m.into_iter().collect::<Vec<_>>();
        assert_eq!(got, [(8, 42), (9, 42), (10, 99), (11, 42), (12, 42)]);
    }

    #[test]
    fn test_entries_size_hint_progress_and_fused_behavior() {
        let mut m: IndexMap<u64, u64> = IndexMap::with_capacity_for_range(0..=4);
        m.insert(1, 1);
        m.insert(3, 3);
        let mut it = m.entries(0..=4);
        let (lo, hi) = it.size_hint();
        // Total slots in range are 5; EntriesIter yields 5 items (occupied or vacant)
        assert_eq!(lo, 5);
        assert_eq!(hi, Some(5));
        // Consume one and check size_hint decreases
        let _ = it.next();
        let (lo2, hi2) = it.size_hint();
        assert_eq!(lo2, 4);
        assert_eq!(hi2, Some(4));
        // Drain remaining
        for _ in it {}
        // Further nexts should be None (Fused semantics)
        let mut it2 = m.entries(0..=4);
        for _ in 0..5 {
            let _ = it2.next();
        }
        assert!(it2.next().is_none());
        assert!(it2.next().is_none());
    }

    #[test]
    fn test_len_updates_with_entries_insert_and_remove() {
        let mut m: IndexMap<u64, u64> = IndexMap::with_capacity_for_range(0..=2);
        assert_eq!(m.len(), 0);
        m.entries(0..=2).for_each(|e| match e {
            IndexMapEntry::Vacant(v) => {
                v.insert(1);
            }
            IndexMapEntry::Occupied(_) => unreachable!(),
        });
        assert_eq!(m.len(), 3);
        // Remove the middle via entry
        match m.entry(1) {
            IndexMapEntry::Occupied(o) => {
                let _ = o.remove();
            }
            _ => unreachable!(),
        }
        assert_eq!(m.len(), 2);
        // Filling again should only add back one
        m.entries(0..=2).for_each(|e| match e {
            IndexMapEntry::Vacant(v) => {
                v.insert(2);
            }
            IndexMapEntry::Occupied(_) => {}
        });
        assert_eq!(m.len(), 3);
    }

    #[test]
    fn test_entries_from_iter_empty_is_empty_and_noop() {
        let mut m: IndexMap<u64, u64> = IndexMap::new();
        m.insert(10, 1);
        let len_before = m.len();
        let empty: [u64; 0] = [];
        let mut it = m.entries_from_iter(&empty);
        assert_eq!(it.size_hint(), (0, Some(0)));
        assert!(it.next().is_none());
        assert_eq!(m.len(), len_before);
        assert_eq!(m.into_iter().collect::<Vec<_>>(), [(10, 1)]);
    }

    #[test]
    fn test_entries_from_iter_spans_min_max_and_mutates() {
        let mut m: IndexMap<u64, u64> = IndexMap::new();
        m.insert(3, 3);
        let keys = [5u64, 2u64, 4u64];
        m.entries_from_iter(&keys).for_each(|entry| match entry {
            IndexMapEntry::Occupied(mut o) => {
                let k = *o.key();
                o.insert(k * 10 + 1);
            }
            IndexMapEntry::Vacant(v) => {
                let k = *v.key();
                v.insert(k * 10);
            }
        });
        // Range should be 2..=5
        assert_eq!(
            m.into_iter().collect::<Vec<_>>(),
            [(2, 20), (3, 31), (4, 40), (5, 50)]
        );
    }

    #[test]
    fn test_entries_from_iter_with_duplicates_yields_one_slot() {
        let mut m: IndexMap<u64, u64> = IndexMap::new();
        let keys = [4u64, 4u64, 4u64];
        let mut it = m.entries_from_iter(&keys);
        // Range is 4..=4 → exactly one element in iterator
        assert_eq!(it.size_hint(), (1, Some(1)));
        let first = it.next();
        assert!(first.is_some());
        assert!(it.next().is_none());
    }
}

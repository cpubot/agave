use {
    bytemuck::{Pod, Zeroable},
    num_traits::{ConstOne, ConstZero, PrimInt},
    std::{
        iter::Enumerate,
        ops::{BitAndAssign, Bound, RangeBounds},
        slice::Iter,
    },
    thiserror::Error,
};

#[derive(Debug, Error, PartialEq, Eq)]
pub enum Error {
    #[error("index out of bounds")]
    OutOfBounds,
}

/// A fixed-size bit array backed by an array of integer words.
///
/// Provides efficient storage and manipulation of boolean flags using bit
/// operations. The total capacity is `NUM_WORDS * std::mem::size_of::<Word>() *
/// 8` bits.
///
/// # Type Parameters
///
/// * `Word` - The integer type used for storage
/// * `NUM_WORDS` - Number of words in the array
///
/// # Examples
///
/// Basic usage:
/// ```
/// # use solana_ledger::bit_array::*;
/// let mut bits = BitArray::<u64, 1>::default();
/// bits.insert(0);
/// bits.insert(63);
/// assert!(bits.contains(0));
/// assert!(bits.contains(63));
/// assert!(!bits.contains(32));
/// ```
///
/// Using ranges:
/// ```
/// # use solana_ledger::bit_array::*;
/// # fn main() -> Result<(), Error> {
/// let mut bits = BitArray::<u64, 1>::default();
/// bits.insert(1);
/// bits.insert(2);
/// assert_eq!(bits.range(1..3)?.indices().collect::<Vec<_>>(), vec![1, 2]);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
#[repr(transparent)]
pub struct BitArray<Word, const NUM_WORDS: usize> {
    pub index: [Word; NUM_WORDS],
}

impl<Word, const NUM_WORDS: usize> Default for BitArray<Word, NUM_WORDS>
where
    Word: ConstZero,
{
    fn default() -> Self {
        Self {
            index: [Word::ZERO; NUM_WORDS],
        }
    }
}

impl<Word, const NUM_WORDS: usize> BitArray<Word, NUM_WORDS> {
    pub const WORD_BITS: usize = std::mem::size_of::<Word>() * 8;
    pub const MAX_INDEX: usize = NUM_WORDS * Self::WORD_BITS;

    /// Get the word and bit offset for the given index.
    ///
    /// # Examples
    ///
    /// ```
    /// # use solana_ledger::bit_array::*;
    /// let (word_idx, bit_idx) = BitArray::<u64, 16>::location_of(63);
    /// assert_eq!(word_idx, 0);
    /// assert_eq!(bit_idx, 63);
    /// ```
    pub fn location_of(idx: usize) -> (usize, usize) {
        let word_idx = idx / Self::WORD_BITS;
        let bit_idx = idx % Self::WORD_BITS;
        (word_idx, bit_idx)
    }

    fn assert_bounds(&self, idx: usize) -> Result<(), Error> {
        if idx >= Self::MAX_INDEX {
            return Err(Error::OutOfBounds);
        }
        Ok(())
    }
}

impl<Word, const NUM_WORDS: usize> BitArray<Word, NUM_WORDS>
where
    Word: PrimInt + ConstOne + ConstZero + BitAndAssign,
{
    /// Remove a bit at the given index.
    ///
    /// Returns `true` if the bit was set, `false` otherwise.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    ///
    /// ```should_panic
    /// # use solana_ledger::bit_array::*;
    /// let mut bit_array = BitArray::<u64, 1>::default();
    /// bit_array.remove_unchecked(64);
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// # use solana_ledger::bit_array::*;
    /// let mut bit_array = BitArray::<u64, 1>::default();
    /// assert!(bit_array.insert_unchecked(63));
    /// assert!(bit_array.remove_unchecked(63));
    /// assert!(!bit_array.remove_unchecked(63));
    /// ```
    pub fn remove_unchecked(&mut self, idx: usize) -> bool {
        let (word_idx, bit_idx) = Self::location_of(idx);
        let prev = self.index[word_idx];
        let next = prev & !(Word::ONE << bit_idx);
        if prev != next {
            self.index[word_idx] = next;
            true
        } else {
            false
        }
    }

    /// Remove a bit at the given index.
    ///
    /// Returns `true` if the bit was set, `false` otherwise.
    ///
    /// # Errors
    ///
    /// Returns an error if the index is out of bounds.
    ///
    /// ```
    /// # use solana_ledger::bit_array::*;
    /// let mut bit_array = BitArray::<u64, 1>::default();
    /// assert!(bit_array.remove(64).is_err());
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// # use solana_ledger::bit_array::*;
    /// let mut bit_array = BitArray::<u64, 1>::default();
    /// assert!(bit_array.insert_unchecked(63));
    /// assert!(bit_array.remove(63).is_ok());
    /// assert!(!bit_array.remove(63).unwrap());
    /// ```
    pub fn remove(&mut self, idx: usize) -> Result<bool, Error> {
        self.assert_bounds(idx)?;
        Ok(self.remove_unchecked(idx))
    }

    /// Insert a bit at the given index.
    ///
    /// Returns `true` if the bit was not already set, `false` otherwise.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    ///
    /// ```should_panic
    /// # use solana_ledger::bit_array::*;
    /// let mut bit_array = BitArray::<u64, 1>::default();
    /// bit_array.insert_unchecked(64);
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// # use solana_ledger::bit_array::*;
    /// let mut bit_array = BitArray::<u64, 1>::default();
    /// assert!(bit_array.insert_unchecked(63));
    /// assert!(!bit_array.insert_unchecked(63));
    /// ```
    pub fn insert_unchecked(&mut self, idx: usize) -> bool {
        let (word_idx, bit_idx) = Self::location_of(idx);
        let prev = self.index[word_idx];
        let next = prev | (Word::ONE << bit_idx);
        if prev != next {
            self.index[word_idx] = next;
            true
        } else {
            false
        }
    }

    /// Insert a bit at the given index.
    ///
    /// Returns `true` if the bit was not already set, `false` otherwise.
    ///
    /// # Errors
    ///
    /// Returns an error if the index is out of bounds.
    ///
    /// ```
    /// # use solana_ledger::bit_array::*;
    /// let mut bit_array = BitArray::<u64, 1>::default();
    /// assert!(bit_array.insert(64).is_err());
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// # use solana_ledger::bit_array::*;
    /// let mut bit_array = BitArray::<u64, 1>::default();
    /// assert!(bit_array.insert(63).is_ok());
    /// assert!(!bit_array.insert(63).unwrap());
    /// ```
    pub fn insert(&mut self, idx: usize) -> Result<bool, Error> {
        self.assert_bounds(idx)?;
        Ok(self.insert_unchecked(idx))
    }

    /// Check if a bit is set at the given index.
    ///
    /// # Examples
    ///
    /// ```
    /// # use solana_ledger::bit_array::*;
    /// let mut bit_array = BitArray::<u64, 1>::default();
    /// bit_array.insert(63);
    /// assert!(bit_array.contains(63));
    /// ```
    pub fn contains(&self, idx: usize) -> bool {
        if self.assert_bounds(idx).is_err() {
            return false;
        }

        let (word_idx, bit_idx) = Self::location_of(idx);
        (self.index[word_idx] & (Word::ONE << bit_idx)) != Word::ZERO
    }

    /// Get an iterator over the words in the array.
    ///
    /// See [`BitArraySlice`] for more information.
    pub fn iter(&self) -> BitArraySlice<Word, NUM_WORDS> {
        BitArraySlice::from_range_unchecked(self, ..)
    }

    /// Get an iterator over the indices of the set bits in the array.
    ///
    /// See [`BitArraySlice::indices`] for more information.
    ///
    /// # Examples
    ///
    /// ```
    /// # use solana_ledger::bit_array::*;
    /// let mut bit_array = BitArray::<u64, 1>::default();
    /// bit_array.insert(0);
    /// bit_array.insert(1);
    /// assert_eq!(bit_array.indices().collect::<Vec<_>>(), [0, 1]);
    /// ```
    pub fn indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.iter().indices()
    }

    /// Count the number of set bits in the array.
    ///
    /// See [`BitArraySlice::count_ones`] for more information.
    ///
    /// # Examples
    ///
    /// ```
    /// # use solana_ledger::bit_array::*;
    /// let mut bit_array = BitArray::<u64, 1>::default();
    /// bit_array.insert(0);
    /// bit_array.insert(1);
    /// assert_eq!(bit_array.count_ones(), 2);
    /// ```
    pub fn count_ones(&self) -> usize {
        self.iter().count_ones()
    }

    /// Get an iterator over the bits in the array within the given range.
    ///
    /// See [`BitArraySlice::from_range`] for more information.
    ///
    /// # Errors
    ///
    /// Returns an error if the range is out of bounds.
    ///
    /// ```
    /// # use solana_ledger::bit_array::*;
    /// let bit_array = BitArray::<u64, 1>::default();
    /// assert!(bit_array.range(..64).is_ok());
    /// assert!(bit_array.range(64..).is_err());
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// # use solana_ledger::bit_array::*;
    ///
    /// # fn main() -> Result<(), Error> {
    /// let mut bit_array = BitArray::<u64, 1>::default();
    /// bit_array.insert(0);
    /// bit_array.insert(1);
    /// assert_eq!(bit_array.range(..2)?.indices().collect::<Vec<_>>(), [0, 1]);
    /// assert_eq!(bit_array.range(1..)?.count_ones(), 1);
    /// # Ok(())
    /// # }
    /// ```
    pub fn range(
        &self,
        bounds: impl RangeBounds<usize>,
    ) -> Result<BitArraySlice<Word, NUM_WORDS>, Error> {
        BitArraySlice::from_range(self, bounds)
    }

    /// Get an iterator over the bits in the array within the given range.
    ///
    /// See [`BitArraySlice::from_range_unchecked`] for more information.
    ///
    /// # Panics
    ///
    /// Panics if the range is out of bounds.
    ///
    /// ```should_panic
    /// # use solana_ledger::bit_array::*;
    /// let mut bit_array = BitArray::<u64, 1>::default();
    /// bit_array.range_unchecked(..65);
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// # use solana_ledger::bit_array::*;
    /// let mut bit_array = BitArray::<u64, 1>::default();
    /// bit_array.insert(0);
    /// bit_array.insert(1);
    /// assert_eq!(bit_array.range_unchecked(..2).indices().collect::<Vec<_>>(), [0, 1]);
    /// assert_eq!(bit_array.range_unchecked(1..).count_ones(), 1);
    /// ```
    pub fn range_unchecked(
        &self,
        bounds: impl RangeBounds<usize>,
    ) -> BitArraySlice<Word, NUM_WORDS> {
        BitArraySlice::from_range_unchecked(self, bounds)
    }
}

/// A slice of a [`BitArray`] that provides efficient bit-level access and
/// operations.
///
/// This struct is used to iterate over a subset of the bits in a [`BitArray`],
/// providing methods for counting set bits and iterating over set indices.
///
/// # Type Parameters
///
/// * `Word` - The integer type used for storage
/// * `NUM_WORDS` - Number of words in the array
///
/// # Iteration
///
/// When iterating over a `BitArraySlice`, it yields tuples of `(usize, Word)`
/// where:
/// - The `usize` represents the starting bit position of the current word. For
///   example, with 64-bit words:
///   - Word 0 starts at bit index 0.
///   - Word 1 starts at bit index 64.
///   - Word 2 starts at bit index 128.
/// - The `Word` contains the masked bits for each word in the current slice
///   segment. The word is masked to only include bits that fall within the
///   slice bounds. For example:
///   - If the slice is `0..64`, the word will contain bits 0 to 63.
///   - If the slice is `60..100`, the first word will only contain bits 60 to
///     63 (masking off bits 0-59), and the second word will only contain bits
///     64 to 99 (masking off bits 100-127).
pub struct BitArraySlice<'a, Word, const NUM_WORDS: usize> {
    start: usize,
    end: usize,
    start_word: usize,
    iter: Enumerate<Iter<'a, Word>>,
}

struct BitArraySliceBounds {
    /// The start index of the requested slice.
    start: usize,
    /// The end index of the requested slice.
    end: usize,
    /// The index of the first word in the slice given the start index.
    start_word: usize,
    /// The index of the last word in the slice given the end index.
    end_word: usize,
}

impl<'a, Word, const NUM_WORDS: usize> BitArraySlice<'a, Word, NUM_WORDS> {
    const WORD_BITS: usize = BitArray::<Word, NUM_WORDS>::WORD_BITS;
    const MAX_INDEX: usize = BitArray::<Word, NUM_WORDS>::MAX_INDEX;

    /// Compute the bounds for a slice of a [`BitArray`] given a range.
    ///
    /// # Bounds
    /// - `start` is the start index of the requested slice.
    /// - `end` is the end index of the requested slice.
    /// - `start_word` is the index of the first word in the slice given the
    ///   start index.
    /// - `end_word` is the index of the last word in the slice given the end
    ///   index.
    ///
    /// `start` and `end` are normalized to [`Bound::Included`] and
    /// [`Bound::Excluded`] bounds, i.e., `start..end`.
    fn compute_bounds<R>(bounds: R) -> BitArraySliceBounds
    where
        R: RangeBounds<usize>,
    {
        let start = match bounds.start_bound() {
            Bound::Included(&n) => n,
            Bound::Excluded(&n) => n + 1,
            Bound::Unbounded => 0,
        };
        let end = match bounds.end_bound() {
            Bound::Included(&n) => n + 1,
            Bound::Excluded(&n) => n,
            Bound::Unbounded => Self::MAX_INDEX,
        };
        let end_word: usize = end.div_ceil(Self::WORD_BITS);
        let start_word = start / Self::WORD_BITS;

        BitArraySliceBounds {
            start,
            end,
            start_word,
            end_word,
        }
    }

    /// Create a new [`BitArraySlice`] from a range without bounds checking.
    ///
    /// # Panics
    ///
    /// Panics if the range is out of bounds.
    pub fn from_range_unchecked<R>(bit_array: &'a BitArray<Word, NUM_WORDS>, bounds: R) -> Self
    where
        R: RangeBounds<usize>,
    {
        let BitArraySliceBounds {
            start,
            end,
            start_word,
            end_word,
        } = Self::compute_bounds(bounds);

        Self {
            start,
            end,
            start_word,
            iter: bit_array.index[start_word..end_word].iter().enumerate(),
        }
    }

    /// Create a new [`BitArraySlice`] from a range with bounds checking.
    ///
    /// # Errors
    ///
    /// Returns an error if the range is out of bounds.
    pub fn from_range<R>(bit_array: &'a BitArray<Word, NUM_WORDS>, bounds: R) -> Result<Self, Error>
    where
        R: RangeBounds<usize>,
    {
        let BitArraySliceBounds {
            start,
            end,
            start_word,
            end_word,
        } = Self::compute_bounds(bounds);

        if start >= Self::MAX_INDEX || end > Self::MAX_INDEX {
            return Err(Error::OutOfBounds);
        }

        Ok(Self {
            start,
            end,
            start_word,
            iter: bit_array.index[start_word..end_word].iter().enumerate(),
        })
    }
}

impl<Word, const NUM_WORDS: usize> Iterator for BitArraySlice<'_, Word, NUM_WORDS>
where
    Word: PrimInt + ConstZero,
{
    type Item = (usize, Word);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(word_idx, &word)| {
            // Calculate the absolute bit index for this word:
            // - `self.start_word` is from which word in the original array we started
            // - `word_idx` is how many words we've moved through in this slice
            //
            // Example: If we're slicing starting from the second word (start_word = 1)
            // and we're looking at the first word in our slice (word_idx = 0):
            // `base_idx = (1 + 0) * 64 = 64`.
            let base_idx = (self.start_word + word_idx) * Self::WORD_BITS;

            // Calculate which bits we should keep based on slice bounds.
            //
            // Example: If we started our slice at bit 70, and we're looking at the
            // word that contains bits 64-127, we need to mask off bits 64-69.
            let lower_bound = self.start.saturating_sub(base_idx);
            // Similarly, if our slice ends at bit 100, we need to mask off bits 101-127
            let upper_bound = if base_idx + Self::WORD_BITS > self.end {
                self.end - base_idx
            } else {
                Self::WORD_BITS
            };

            // Create and apply the masks to only keep bits within our slice bounds
            let lower_mask = !Word::ZERO << lower_bound;
            let upper_mask = !Word::ZERO >> (Self::WORD_BITS - upper_bound);
            (base_idx, word & lower_mask & upper_mask)
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<Word, const NUM_WORDS: usize> ExactSizeIterator for BitArraySlice<'_, Word, NUM_WORDS> where
    Word: PrimInt + ConstZero
{
}

impl<Word, const NUM_WORDS: usize> BitArraySlice<'_, Word, NUM_WORDS>
where
    Word: PrimInt + ConstZero,
{
    /// Count the number of set bits in the slice.
    ///
    /// # Examples
    ///
    /// ```
    /// # use solana_ledger::bit_array::*;
    /// # fn main() -> Result<(), Error> {
    /// let mut bit_array = BitArray::<u64, 1>::default();
    /// bit_array.insert(0);
    /// bit_array.insert(1);
    /// assert_eq!(bit_array.range(..32)?.count_ones(), 2);
    /// assert_eq!(bit_array.range(1..32)?.count_ones(), 1);
    /// assert_eq!(bit_array.range(2..32)?.count_ones(), 0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn count_ones(self) -> usize {
        self.map(|(_, mask)| mask.count_ones() as usize).sum()
    }
}

impl<'a, Word, const NUM_WORDS: usize> BitArraySlice<'a, Word, NUM_WORDS>
where
    Word: PrimInt + ConstZero + ConstOne + BitAndAssign,
{
    /// Get an iterator over the indices of the set bits in the slice.
    ///
    /// # Examples
    ///
    /// ```
    /// # use solana_ledger::bit_array::*;
    /// # fn main() -> Result<(), Error> {
    /// let mut bit_array = BitArray::<u64, 1>::default();
    /// bit_array.insert(0);
    /// bit_array.insert(1);
    /// assert_eq!(bit_array.range(..32)?.indices().collect::<Vec<_>>(), [0, 1]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn indices(self) -> impl Iterator<Item = usize> + 'a {
        self.flat_map(|(base_idx, mask)| {
            std::iter::from_fn({
                let mut remaining = mask;
                move || {
                    if remaining == Word::ZERO {
                        None
                    } else {
                        // Find position of lowest set bit.
                        let bit_idx = remaining.trailing_zeros();
                        // Clear the lowest set bit.
                        remaining &= remaining - Word::ONE;
                        // Convert bit position to absolute index by adding word's base_idx.
                        Some(base_idx + bit_idx as usize)
                    }
                }
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use {super::*, proptest::prelude::*};

    proptest! {
        // Property: insert followed by contains should return true
        #[test]
        fn insert_then_contains(idx in 0..BitArray::<u64, 16>::MAX_INDEX) {
            let mut bits = BitArray::<u64, 16>::default();
            prop_assert!(!bits.contains(idx));
            bits.insert_unchecked(idx);
            prop_assert!(bits.contains(idx));
        }

        // Property: insert followed by remove should return true
        #[test]
        fn insert_then_remove(idx in 0..BitArray::<u64, 16>::MAX_INDEX) {
            let mut bits = BitArray::<u64, 16>::default();
            prop_assert!(!bits.remove_unchecked(idx));
            bits.insert_unchecked(idx);
            prop_assert!(bits.remove_unchecked(idx));
        }

        // Property: range queries should return correct indices and counts
        #[test]
        fn range_query_correctness(
            indices in prop::collection::btree_set(
                0..BitArray::<u64, 16>::MAX_INDEX,
                0..BitArray::<u64, 16>::MAX_INDEX
            ),
            // Generate a smaller set of random ranges to test
            ranges in prop::collection::vec(
                (0..BitArray::<u64, 16>::MAX_INDEX)
                    .prop_flat_map(|start| {
                        (start + 1..=BitArray::<u64, 16>::MAX_INDEX)
                            .prop_map(move |end| (start, end))
                    }),
                1..100
            )
        ) {
            let mut bit_array = BitArray::<u64, 16>::default();

            for &idx in &indices {
                bit_array.insert_unchecked(idx);
            }

            for (start, end) in ranges {
                // Test indices match
                assert_eq!(
                    bit_array.range_unchecked(start..end).indices().collect::<Vec<_>>(),
                    indices.range(start..end).copied().collect::<Vec<_>>()
                );

                // Test count matches
                prop_assert_eq!(
                    bit_array.range_unchecked(start..end).count_ones(),
                    indices.range(start..end).count()
                );
            }
        }

        // Property: inserting indices then iterating should return the same indices in order
        #[test]
        fn iter_returns_inserted_indices(
            indices in prop::collection::btree_set(
                0..BitArray::<u64, 16>::MAX_INDEX,
                0..BitArray::<u64, 16>::MAX_INDEX
            )
        ) {
            let mut bit_array = BitArray::<u64, 16>::default();

            for &idx in &indices {
                bit_array.insert_unchecked(idx);
            }

            assert_eq!(
                bit_array.iter().indices().collect::<Vec<_>>(),
                indices.iter().copied().collect::<Vec<_>>()
            );
            prop_assert_eq!(bit_array.iter().count_ones(), indices.len());
            prop_assert_eq!(bit_array.iter().count_ones(), bit_array.iter().indices().count());
        }
    }

    #[test]
    fn test_range_bounds() {
        use std::ops::Bound::*;

        let mut bit_array = BitArray::<u64, 1>::default();
        let input: Vec<_> = (0..10).collect();
        for idx in &input {
            bit_array.insert_unchecked(*idx);
        }

        let cases = [
            ((Unbounded, Excluded(4)), &input[..4]),
            ((Unbounded, Included(3)), &input[..4]),
            ((Unbounded, Unbounded), &input[..]),
            ((Included(2), Excluded(4)), &input[2..4]),
            ((Included(2), Included(3)), &input[2..4]),
            ((Included(2), Unbounded), &input[2..]),
            ((Excluded(2), Unbounded), &input[3..]),
            ((Excluded(2), Excluded(4)), &input[3..4]),
            ((Excluded(2), Included(3)), &input[3..4]),
            // Edge cases
            // Empty range
            ((Included(5), Excluded(5)), &[]),
            // Single element after exclusion
            ((Excluded(4), Included(5)), &input[5..6]),
            // Single element
            ((Included(5), Included(5)), &input[5..6]),
            // Start > End
            ((Included(5), Excluded(3)), &[]),
        ];

        for ((start, end), expected) in cases {
            let result = bit_array
                .range_unchecked((start, end))
                .indices()
                .collect::<Vec<_>>();
            assert_eq!(
                result, expected,
                "Failed for bounds: ({:?}, {:?})",
                start, end
            );
        }
    }

    #[test]
    fn test_range_bounds_errors() {
        use std::ops::Bound::*;

        let mut bit_array = BitArray::<u64, 1>::default();
        let input: Vec<_> = (0..10).collect();
        for idx in &input {
            bit_array.insert_unchecked(*idx);
        }

        let error_cases = [
            // Start beyond array size (64 bits)
            (Included(64), Excluded(65)),
            // End beyond array size
            (Included(0), Excluded(65)),
            // Both beyond array size
            (Included(100), Excluded(200)),
            // Included end at max
            (Unbounded, Included(64)),
            // Included start at max
            (Included(64), Unbounded),
        ];

        for (start, end) in error_cases {
            assert!(
                bit_array.range((start, end)).is_err(),
                "Expected error for bounds: ({:?}, {:?})",
                start,
                end
            );
        }
    }
}

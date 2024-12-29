use {
    crate::{bit_array::BitArray, blockstore::MAX_DATA_SHREDS_PER_SLOT},
    bytemuck::{Pod, Zeroable},
    solana_sdk::clock::Slot,
    static_assertions::const_assert,
};

mod serde;

#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct IndexV2 {
    pub slot: Slot,
    pub(crate) data: ShredIndexV2,
    pub(crate) coding: ShredIndexV2,
}

/// Integer type to house shred indices in the [`BitArray`] of [`ShredIndexV2`].
type Word = u64;

/// Number of `Word`s required to accommodate each shred in a slot ([`MAX_DATA_SHREDS_PER_SLOT`]).
///
/// **THIS VALUE MUST NEVER DECREASE ONCE ROLLED OUT.**
///
/// As it relates to deserializing from the blockstore, using a statically sized structure
/// can trivially accommodate increases in space, but decreases are problematic. For example,
///
/// Increase case:
/// - Assume `Word` is `u64`.
/// - Assume `MAX_DATA_SHREDS_PER_SLOT` was bumped from `32_768` to `65_536`.
/// - Assume data being deserialized was written when `MAX_DATA_SHREDS_PER_SLOT` was `32_768`.
/// - ✅ No issue, as `ShredIndexV2Inner` has extra space that can be zeroed to accommodate.
///
/// Decrease case:
/// - Assume `Word` is `u64`.
/// - Assume `MAX_DATA_SHREDS_PER_SLOT` was reduced from  `65_536` to `32_768`.
/// - Assume data being deserialized was written when `MAX_DATA_SHREDS_PER_SLOT` was `65_536`.
/// - ❌ `ShredIndexV2Inner` only has space to accommodate `32_768` shreds.
///
/// As such, we:
/// - Decouple the definition of `NUM_WORDS_PER_SHRED_INDEX_V2` from `MAX_DATA_SHREDS_PER_SLOT`.
///   (i.e., make it not an explicit function of)
/// - Impose a compile time check to ensure that `NUM_WORDS_PER_SHRED_INDEX_V2 >= MAX_DATA_SHREDS_PER_SLOT.div_ceil(64)`.
/// - Impose a convention that `NUM_WORDS_PER_SHRED_INDEX_V2` must never decrease.
const NUM_WORDS_PER_SHRED_INDEX_V2: usize = 512;
pub type ShredIndexV2Inner = BitArray<Word, NUM_WORDS_PER_SHRED_INDEX_V2>;
const_assert!(ShredIndexV2Inner::MAX_INDEX >= MAX_DATA_SHREDS_PER_SLOT);
const SIZE_OF_INDEX: usize = std::mem::size_of::<ShredIndexV2Inner>();

/// A bit array of shred indices, where each u64 represents 64 shred indices.
///
/// The current implementation of [`ShredIndex`] utilizes a [`BTreeSet`] to store
/// shred indices. While [`BTreeSet`] remains efficient as operations are amortized
/// over time, the overhead of the B-tree structure becomes significant when frequently
/// serialized and deserialized. In particular:
/// - **Tree Traversal**: Serialization requires walking the non-contiguous tree structure.
/// - **Reconstruction**: Deserialization involves rebuilding the tree in bulk,
///   including dynamic memory allocations and re-balancing nodes.
///
/// In contrast, our bit array implementation provides:
/// - **Contiguous Memory**: All bits are stored in a contiguous array of u64 words,
///   allowing direct indexing and efficient memory access patterns.
/// - **Direct Range Access**: Can load only the specific words that overlap with a
///   requested range, avoiding unnecessary traversal.
/// - **Simplified Serialization**: The contiguous memory layout allows for efficient
///   serialization/deserialization without tree reconstruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
#[repr(C)]
pub struct ShredIndexV2 {
    pub(crate) size_of_index: usize,
    pub(crate) num_shreds: usize,
    pub(crate) index: ShredIndexV2Inner,
}

impl Default for ShredIndexV2 {
    fn default() -> Self {
        Self {
            size_of_index: SIZE_OF_INDEX,
            num_shreds: 0,
            index: ShredIndexV2Inner::default(),
        }
    }
}

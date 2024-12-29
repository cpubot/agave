use {
    super::*,
    crate::to_from_bytes::{cursor::SliceCursor, Error, Result, ToFromBytes},
    std::{mem::MaybeUninit, ptr::addr_of_mut},
};

impl ShredIndexV2 {
    #[cfg(not(target_endian = "little"))]
    #[inline(always)]
    fn as_uninit(&self, arr: *mut Self) {
        unsafe {
            addr_of_mut!((*arr).size_of_index).write(self.size_of_index.to_le());
            addr_of_mut!((*arr).num_shreds).write(self.num_shreds.to_le());
            let dst = addr_of_mut!((*arr).index) as *mut Word;
            for (i, &word) in self.index.index.iter().enumerate() {
                dst.add(i).write(word.to_le());
            }
        }
    }

    #[inline(always)]
    fn from_bytes_into_uninit(bytes: &mut SliceCursor<u8>, ptr: *mut Self) -> Result<()> {
        let size_of_index = bytes.read_usize()?;
        let num_shreds = bytes.read_usize()?;
        let src = bytes.read_exact(size_of_index)?;

        if src.len() != size_of_index {
            return Err(Error::InvalidLength);
        }

        unsafe {
            addr_of_mut!((*ptr).size_of_index).write(SIZE_OF_INDEX);
            addr_of_mut!((*ptr).num_shreds).write(num_shreds);

            #[cfg(not(target_endian = "little"))]
            {
                let dst = addr_of_mut!((*ptr).index) as *mut Word;
                for (i, word) in src.chunks_exact(8).enumerate() {
                    dst.add(i)
                        .write(Word::from_le_bytes(word.try_into().unwrap_unchecked()));
                }
            }

            let dst = addr_of_mut!((*ptr).index) as *mut u8;

            #[cfg(target_endian = "little")]
            {
                dst.copy_from_nonoverlapping(src.as_ptr(), size_of_index);
            }

            dst.add(size_of_index)
                .write_bytes(0, SIZE_OF_INDEX - size_of_index);
        }

        Ok(())
    }
}

impl ToFromBytes for IndexV2 {
    #[inline(always)]
    #[cfg(not(target_endian = "little"))]
    fn to_bytes<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce(&[u8]) -> R,
    {
        let mut slf = MaybeUninit::<Self>::uninit();
        let ptr = slf.as_mut_ptr();

        unsafe {
            addr_of_mut!((*ptr).slot).write(self.slot.to_le());
            self.data.as_uninit(addr_of_mut!((*ptr).data));
            self.coding.as_uninit(addr_of_mut!((*ptr).coding));

            Ok(f(std::slice::from_raw_parts(
                slf.as_ptr() as *const u8,
                std::mem::size_of::<Self>(),
            )))
        }
    }

    #[inline(always)]
    #[cfg(target_endian = "little")]
    fn to_bytes<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce(&[u8]) -> R,
    {
        Ok(f(bytemuck::bytes_of(self)))
    }

    #[inline(always)]
    fn from_bytes(bytes: &[u8]) -> Result<Self> {
        #[cfg(target_endian = "little")]
        {
            if let Ok(slice) = bytemuck::try_from_bytes(bytes) {
                return Ok(*slice);
            }
        }

        if bytes.len() > std::mem::size_of::<Self>() {
            return Err(Error::TooManyBytes);
        }

        let mut slf = MaybeUninit::<Self>::uninit();
        let mut cursor = SliceCursor::new(bytes);
        let slot = cursor.read_u64()?;
        let ptr = slf.as_mut_ptr();

        unsafe {
            addr_of_mut!((*ptr).slot).write(slot);

            ShredIndexV2::from_bytes_into_uninit(&mut cursor, addr_of_mut!((*ptr).data))?;
            ShredIndexV2::from_bytes_into_uninit(&mut cursor, addr_of_mut!((*ptr).coding))?;

            Ok(slf.assume_init())
        }
    }
}

#[cfg(test)]
mod tests {
    use {
        super::*,
        crate::blockstore_meta::{IndexLegacy, ShredIndexLegacy},
        proptest::prelude::*,
        std::collections::BTreeSet,
    };

    trait ShredIndex {
        fn insert(&mut self, index: u64);
    }

    trait Index {
        type ShredIndex: ShredIndex;
        fn new(slot: u64) -> Self;
        fn data_mut(&mut self) -> &mut Self::ShredIndex;
        fn coding_mut(&mut self) -> &mut Self::ShredIndex;
    }

    impl ShredIndex for ShredIndexV2 {
        #[inline(always)]
        fn insert(&mut self, index: u64) {
            self.insert(index);
        }
    }

    impl Index for IndexV2 {
        type ShredIndex = ShredIndexV2;

        #[inline(always)]
        fn new(slot: u64) -> Self {
            Self::new(slot)
        }

        #[inline(always)]
        fn data_mut(&mut self) -> &mut Self::ShredIndex {
            self.data_mut()
        }

        #[inline(always)]
        fn coding_mut(&mut self) -> &mut Self::ShredIndex {
            self.coding_mut()
        }
    }

    impl ShredIndex for ShredIndexLegacy {
        #[inline(always)]
        fn insert(&mut self, index: u64) {
            self.insert(index);
        }
    }

    impl Index for IndexLegacy {
        type ShredIndex = ShredIndexLegacy;

        #[inline(always)]
        fn new(slot: u64) -> Self {
            Self::new(slot)
        }

        #[inline(always)]
        fn data_mut(&mut self) -> &mut Self::ShredIndex {
            self.data_mut()
        }

        #[inline(always)]
        fn coding_mut(&mut self) -> &mut Self::ShredIndex {
            self.coding_mut()
        }
    }

    const SMALLER_INDEX_FACTOR: usize = 2;
    /// A smaller version of [`ShredIndexV2Inner`] that is half the size to emulate the scenario where
    /// MAX_DATA_SHREDS_PER_SLOT increased but the blockstore still contains data from the old
    /// MAX_DATA_SHREDS_PER_SLOT.
    type SmallerShredIndexV2Inner =
        BitArray<Word, { NUM_WORDS_PER_SHRED_INDEX_V2 / SMALLER_INDEX_FACTOR }>;

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
    #[repr(C)]
    pub struct SmallerShredIndexV2 {
        pub(crate) size_of_index: usize,
        pub(crate) num_shreds: usize,
        pub(crate) index: SmallerShredIndexV2Inner,
    }

    impl ShredIndex for SmallerShredIndexV2 {
        #[inline(always)]
        fn insert(&mut self, index: u64) {
            if let Ok(true) = self.index.insert(index as usize) {
                self.num_shreds += 1;
            }
        }
    }

    impl Default for SmallerShredIndexV2 {
        #[inline(always)]
        fn default() -> Self {
            Self {
                size_of_index: std::mem::size_of::<SmallerShredIndexV2Inner>(),
                num_shreds: 0,
                index: SmallerShredIndexV2Inner::default(),
            }
        }
    }

    #[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
    #[repr(C)]
    pub struct SmallerIndexV2 {
        pub slot: Slot,
        pub(crate) data: SmallerShredIndexV2,
        pub(crate) coding: SmallerShredIndexV2,
    }

    impl ToFromBytes for SmallerIndexV2 {
        #[inline(always)]
        fn to_bytes<F, R>(&self, f: F) -> Result<R>
        where
            F: FnOnce(&[u8]) -> R,
        {
            Ok(f(bytemuck::bytes_of(self)))
        }

        #[inline(always)]
        fn from_bytes(bytes: &[u8]) -> Result<Self> {
            Ok(*bytemuck::try_from_bytes(bytes).unwrap())
        }
    }

    impl Index for SmallerIndexV2 {
        type ShredIndex = SmallerShredIndexV2;

        #[inline(always)]
        fn new(slot: u64) -> Self {
            Self {
                slot,
                data: SmallerShredIndexV2::default(),
                coding: SmallerShredIndexV2::default(),
            }
        }

        #[inline(always)]
        fn data_mut(&mut self) -> &mut Self::ShredIndex {
            &mut self.data
        }

        #[inline(always)]
        fn coding_mut(&mut self) -> &mut Self::ShredIndex {
            &mut self.coding
        }
    }

    #[inline(always)]
    fn init_index<I: Index>(
        coding_indices: &BTreeSet<usize>,
        data_indices: &BTreeSet<usize>,
        slot: u64,
    ) -> I {
        let mut index = I::new(slot);
        for &idx in coding_indices {
            index.coding_mut().insert(idx as u64);
        }
        for &idx in data_indices {
            index.data_mut().insert(idx as u64);
        }
        index
    }

    proptest! {
        #[test]
        fn test_smaller_index_compat(
            coding_indices in prop::collection::btree_set(
                0..MAX_DATA_SHREDS_PER_SLOT / SMALLER_INDEX_FACTOR,
                0..MAX_DATA_SHREDS_PER_SLOT / SMALLER_INDEX_FACTOR
            ),
            data_indices in prop::collection::btree_set(
                0..MAX_DATA_SHREDS_PER_SLOT / SMALLER_INDEX_FACTOR,
                0..MAX_DATA_SHREDS_PER_SLOT / SMALLER_INDEX_FACTOR
            ),
            slot in 0..u64::MAX
        ) {
            let smaller_index = init_index::<SmallerIndexV2>(&coding_indices, &data_indices, slot);
            let index = init_index::<IndexV2>(&coding_indices, &data_indices, slot);
            let index_from_smaller = smaller_index.to_bytes(IndexV2::from_bytes).unwrap().unwrap();
            prop_assert_eq!(index, index_from_smaller);
        }

        #[test]
        fn test_serde(
            coding_indices in prop::collection::btree_set(
                0..MAX_DATA_SHREDS_PER_SLOT,
                0..MAX_DATA_SHREDS_PER_SLOT
            ),
            data_indices in prop::collection::btree_set(
                0..MAX_DATA_SHREDS_PER_SLOT,
                0..MAX_DATA_SHREDS_PER_SLOT
            ),
            slot in 0..u64::MAX
        ) {
            let index = init_index::<IndexV2>(&coding_indices, &data_indices, slot);
            let index2 = index.to_bytes(IndexV2::from_bytes).unwrap().unwrap();
            prop_assert_eq!(index, index2);
        }

        #[test]
        fn test_legacy_collision(
            coding_indices in prop::collection::btree_set(
                0..MAX_DATA_SHREDS_PER_SLOT,
                0..MAX_DATA_SHREDS_PER_SLOT
            ),
            data_indices in prop::collection::btree_set(
                0..MAX_DATA_SHREDS_PER_SLOT,
                0..MAX_DATA_SHREDS_PER_SLOT
            ),
            slot in 0..u64::MAX
        ) {
            let index = init_index::<IndexV2>(&coding_indices, &data_indices, slot);
            let legacy = index.to_bytes(IndexLegacy::from_bytes).unwrap();
            prop_assert!(legacy.is_err());
        }

        #[test]
        fn test_legacy_collision_inverse(
            coding_indices in prop::collection::btree_set(
                0..MAX_DATA_SHREDS_PER_SLOT,
                0..MAX_DATA_SHREDS_PER_SLOT
            ),
            data_indices in prop::collection::btree_set(
                0..MAX_DATA_SHREDS_PER_SLOT,
                0..MAX_DATA_SHREDS_PER_SLOT
            ),
            slot in 0..u64::MAX
        ) {
            let index = init_index::<IndexLegacy>(&coding_indices, &data_indices, slot);
            prop_assert_eq!(&index, &index.to_bytes(IndexLegacy::from_bytes).unwrap().unwrap());
            let v2 = index.to_bytes(IndexV2::from_bytes).unwrap();
            prop_assert!(v2.is_err());
        }
    }
}

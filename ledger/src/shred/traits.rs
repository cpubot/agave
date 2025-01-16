use {
    crate::shred::{CodingShredHeader, DataShredHeader, Error, ShredCommonHeader},
    solana_sdk::{clock::Slot, signature::Signature},
    std::borrow::Cow,
};

pub(super) trait Shred<'a>: Sized {
    // Total size of payload including headers, merkle
    // branches (if any), zero paddings, etc.
    const SIZE_OF_PAYLOAD: usize;
    // Size of common and code/data headers.
    const SIZE_OF_HEADERS: usize;

    type SignedData: AsRef<[u8]>;

    fn from_payload(shred: Cow<'a, [u8]>) -> Result<Self, Error>;
    fn common_header(&self) -> &ShredCommonHeader;
    fn sanitize(&self) -> Result<(), Error>;

    fn set_signature(&mut self, signature: Signature);

    fn payload(&self) -> &[u8];
    fn into_payload(self) -> Vec<u8>;

    // Returns the shard index within the erasure coding set.
    fn erasure_shard_index(&self) -> Result<usize, Error>;
    // Returns the portion of the shred's payload which is erasure coded.
    fn erasure_shard(self) -> Result<Vec<u8>, Error>;
    // Like Shred::erasure_shard but returning a slice.
    fn erasure_shard_as_slice(&self) -> Result<&[u8], Error>;

    // Portion of the payload which is signed.
    fn signed_data(&'a self) -> Result<Self::SignedData, Error>;

    // Only for tests.
    fn set_index(&mut self, index: u32);
    fn set_slot(&mut self, slot: Slot);

    /// see: https://github.com/solana-labs/solana/pull/10109
    fn extract_payload(payload: Cow<'a, [u8]>) -> Cow<'a, [u8]> {
        match payload {
            Cow::Borrowed(slice) => Cow::Borrowed(&slice[..Self::SIZE_OF_PAYLOAD]),
            Cow::Owned(mut vec) => {
                vec.truncate(Self::SIZE_OF_PAYLOAD);
                Cow::Owned(vec)
            }
        }
    }
}

pub(super) trait ShredData<'a>: Shred<'a> {
    fn data_header(&self) -> &DataShredHeader;

    fn parent(&self) -> Result<Slot, Error> {
        let slot = self.common_header().slot;
        let parent_offset = self.data_header().parent_offset;
        if parent_offset == 0 && slot != 0 {
            return Err(Error::InvalidParentOffset {
                slot,
                parent_offset,
            });
        }
        slot.checked_sub(Slot::from(parent_offset))
            .ok_or(Error::InvalidParentOffset {
                slot,
                parent_offset,
            })
    }

    fn data(&self) -> Result<&[u8], Error>;
}

pub(super) trait ShredCode<'a>: Shred<'a> {
    fn coding_header(&self) -> &CodingShredHeader;

    fn first_coding_index(&self) -> Option<u32> {
        let position = u32::from(self.coding_header().position);
        self.common_header().index.checked_sub(position)
    }
}

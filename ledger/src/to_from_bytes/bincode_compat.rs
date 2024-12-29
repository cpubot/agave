use super::*;

impl<T> ToFromBytes for T
where
    T: Serialize + DeserializeOwned,
{
    #[inline(always)]
    fn to_bytes<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce(&[u8]) -> R,
    {
        Ok(f(
            &bincode::serialize(self).map_err(|_| Error::BincodeSerialize)?
        ))
    }

    #[inline(always)]
    fn from_bytes(bytes: &[u8]) -> Result<Self> {
        bincode::deserialize(bytes).map_err(|_| Error::BincodeDeserialize)
    }
}

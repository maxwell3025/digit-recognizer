#[derive(Debug)]
pub struct Image{
    pub data: Vec<u8>
}

impl Image{
    pub fn new() -> Self{
        Image{
            data: vec![0; 784]
        }
    }
}

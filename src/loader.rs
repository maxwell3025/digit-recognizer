use std::io::Error;
use std::io::ErrorKind;

pub fn load_image_set(file: String) -> Result<Vec<crate::image::Image>, std::io::Error>{
    let image_data = std::fs::read(file)?;

    if image_data.len() < 8 {
        return Err(Error::new(ErrorKind::InvalidData, "file is too short!"));
    }

    let magic_number: u32 =
    (image_data[0] as u32) << 24|
    (image_data[1] as u32) << 16|
    (image_data[2] as u32) << 8 |
    (image_data[3] as u32) << 0 ;

    if magic_number != 0x803 {
        return Err(Error::new(ErrorKind::InvalidData, format!("invalid header: {:#x}", magic_number)));
    }

    let item_count: u32 =
    (image_data[4] as u32) << 24|
    (image_data[5] as u32) << 16|
    (image_data[6] as u32) << 8 |
    (image_data[7] as u32) << 0 ;
    
    let mut image_list = Vec::new();
    for image_number in 0..item_count as usize {
        let mut image = crate::image::Image::new();
        for image_pixel_index in 0..784 {
            image.data[image_pixel_index] = image_data[image_number * 784 + image_pixel_index + 8];
        }
        image_list.push(image);
    }

    Ok(image_list)
}

pub fn load_label_set(file: String) -> Result<Vec<u8>, Error>{
    let label_data = std::fs::read(file)?;

    if label_data.len() < 8 {
        return Err(Error::new(ErrorKind::InvalidData, "file is too short"));
    }

    let magic_number: u32 =
    (label_data[0] as u32) << 24|
    (label_data[1] as u32) << 16|
    (label_data[2] as u32) << 8 |
    (label_data[3] as u32) << 0 ;

    if magic_number != 0x801 {
        return Err(Error::new(ErrorKind::InvalidData, format!("invalid header: {:#x}", magic_number)));
    }

    let item_count: u32 =
    (label_data[4] as u32) << 24|
    (label_data[5] as u32) << 16|
    (label_data[6] as u32) << 8 |
    (label_data[7] as u32) << 0 ;

    let mut label_list = Vec::new();
    for label_number in 0..item_count as usize {
        label_list.push(label_data[label_number + 8])
    }

    Ok(label_list)
}

use std::fmt;
use std::ops::{Add, Mul, Sub};

#[derive(Debug, Clone)]
pub enum Error {
    NumbersOfElementsError,
    PositionValueError,
    ShapeMismatchError,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
       match *self {
           Error::NumbersOfElementsError => write!(f, "Error: elements mismatch shape"),
           Error::PositionValueError => write!(f, "Error: index out of bounds"),
           Error::ShapeMismatchError => write!(f, "Error: shapes don't match"),
       }
    }
}


// ----------------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}


impl Tensor {
    pub fn new(data: &[f32], shape: &[usize]) -> Result<Tensor, Error> {
        let req_values = shape.iter().product();
        if data.len() != req_values {
            return Err(Error::NumbersOfElementsError);
        }
        Ok(Tensor {
            data: data.to_vec(),
            shape: shape.to_vec(),
        })
    }

    pub fn get_value(&self, coords: &[usize]) -> Result<f32, Error> {
        let shape_size = self.shape.len();
        if coords.len() != shape_size {
            return Err(Error::PositionValueError);
        }
        let row = coords[0];
        let col = coords[1];
        if row >= self.shape[0] || col >= self.shape[1] {
            return Err(Error::ShapeMismatchError);
        }
        let width = self.shape[1];
        let index = (row * width) + col;
        Ok(self.data[index])
    }

    pub fn data(&self) -> &Vec<f32> {
        &self.data
    }

    pub fn add(&self, other_tensor: &Tensor) -> Result<Tensor, Error> {
        if self.shape != other_tensor.shape {
            return Err(Error::ShapeMismatchError);
        }
        let new_tensor_data: Vec<f32> = self.data.iter()
            .zip(other_tensor.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        
        Ok(Tensor { data: new_tensor_data, shape: self.shape.clone() })
    }

    pub fn sub(&self, other_tensor: &Tensor) -> Result<Tensor, Error> {
        if self.shape != other_tensor.shape {
            return Err(Error::ShapeMismatchError);
        }
        let new_tensor_data: Vec<f32> = self.data.iter()
            .zip(other_tensor.data.iter())
            .map(|(a, b)| a - b)
            .collect();
        
        Ok(Tensor { data: new_tensor_data, shape: self.shape.clone() })
    }
    // hadamard / element-wise product
    pub fn mul(&self, other_tensor:&Tensor) -> Result<Tensor, Error> {
        if self.shape != other_tensor.shape {
            return Err(Error::ShapeMismatchError);
        }
        let new_tensor_data:Vec<f32> = self.data.iter()
            .zip(other_tensor.data.iter())
            .map(|(a, b)| a * b)
            .collect();
        Ok(Tensor { data: new_tensor_data, shape: self.shape.clone() })

    }
}

// ----------------------------------------------------------------------------------

impl<'a, 'b> Add<&'b Tensor> for &'a Tensor{
    type Output = Result<Tensor, Error>;
    fn add(self, other_tensor: &'b Tensor) -> Self::Output {
        Tensor::add(self, other_tensor)
    }
}

impl<'a, 'b> Sub<&'b Tensor> for &'a Tensor{
    type Output = Result<Tensor, Error>;
    fn sub(self, other_tensor:&'b Tensor) -> Self::Output {
        Tensor::sub(&self, other_tensor)
    }
}   

impl<'a, 'b> Mul<&'b Tensor> for &'a Tensor{
    type Output = Result<Tensor, Error>;
    fn mul(self, other_tensor:&'b Tensor) -> Self::Output {
        Tensor::mul(&self, other_tensor)
    }
}   

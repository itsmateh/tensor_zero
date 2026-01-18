use std::fmt;
use std::ops::{Add, Mul, Sub};
use rayon::prelude::*;
use rand::Rng;

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

    pub fn zeros(shape:Vec<usize>) -> Tensor {
        let total_size=shape.iter().product();
        Tensor { data: vec![0.0; total_size], shape, }
    }

    pub fn random(shape: Vec<usize>) -> Tensor {
        let total_size: usize = shape.iter().product();
        let data: Vec<f32> = (0..total_size)
            .map(|_| rand::rng().random_range(-1.0..1.0)) 
            .collect();
        Tensor { data, shape }
    }

    fn get_columns(&self) -> usize {
        self.shape[1]
    }
    
    fn get_rows(&self) -> usize {
        self.shape[0]
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
    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
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

    pub fn mul_scalar(&self, scalar:f32) -> Tensor {
        let new_tensor_data:Vec<f32> = self.data.iter()
            .map(|a| a*scalar)
            .collect();
        Tensor{ data:new_tensor_data, shape:self.shape.clone() }
    }

    // pub fn matmul_naive(&self, other_tensor:&Tensor) -> Result<Tensor, Error> {
    //     if (self.shape[1] != other_tensor.shape[0]) || (self.shape.len() != 2 || other_tensor.shape.len() != 2) {
    //         return Err(Error::ShapeMismatchError);
    //     }
    //     let n = self.shape[0]; // A rows
    //     let m = other_tensor.shape[1]; // B columns
    //     let k = self.shape[1]; // or other_tensor.shape[0], the commom dim (A_cols / B_rows)
    //     let mut new_tensor_data:Vec<f32> = vec![0.0;n*m];
    //     // ans[a_rows][b_cols] -> sum(A[a_rows][common_dim], B[common_dim][b_cols])
    //     // row major index = row * total_cols + col
    //     let other_tensor_transpose=other_tensor.transpose();
    //     for i in 0..n{
    //         for j in 0..m{
    //             let mut sum=0.0;
    //             for c in 0..k{
    //                 let a=self.data[(i*k)+c]; 
    //                 let b=other_tensor_transpose.data[(j*k)+c];
    //                 sum+=a*b; 
    //             }
    //             new_tensor_data[(i*m)+j]=sum;
    //         }
    //     }
    //     Ok(Tensor { data:new_tensor_data, shape:vec![n,m]})
    // }

    // pub fn matmul_mem_opt(&self, other_tensor:&Tensor) -> Result<Tensor, Error> {
    //     if (self.shape[1] != other_tensor.shape[0]) || (self.shape.len() != 2 || other_tensor.shape.len() != 2) {
    //         return Err(Error::ShapeMismatchError);
    //     }
    //     let n = self.shape[0]; // A rows
    //     let m = other_tensor.shape[1]; // B columns
    //     let k = self.shape[1]; // or other_tensor.shape[0], the commom dim (A_cols / B_rows)
    //     let mut new_tensor_data:Vec<f32> = vec![0.0;n*m];
    //     // ans[a_rows][b_cols] -> sum(A[a_rows][common_dim], B[common_dim][b_cols])
    //     // row major index = row * total_cols + col
    //     let other_tensor_transpose=other_tensor.transpose();
    //     for i in 0..n{
    //         for j in 0..m{
    //             let mut sum=0.0;
    //             for c in 0..k{
    //                 let a=self.data[(i*k)+c]; 
    //                 let b=other_tensor_transpose.data[(j*k)+c];
    //                 sum+=a*b; 
    //             }
    //             new_tensor_data[(i*m)+j]=sum;
    //         }
    //     }
    //     Ok(Tensor { data:new_tensor_data, shape:vec![n,m]})
    // }
    pub fn matmul(&self, other_tensor:&Tensor) -> Result<Tensor, Error> {
        if (self.shape[1] != other_tensor.shape[0]) || (self.shape.len() != 2 || other_tensor.shape.len() != 2) {
            return Err(Error::ShapeMismatchError);
        }
        let n = self.shape[0]; // A rows
        let m = other_tensor.shape[1]; // B columns
        let k = self.shape[1]; // or other_tensor.shape[0], the commom dim (A_cols / B_rows)
        let mut new_tensor_data:Vec<f32> = vec![0.0;n*m];
        // ans[a_rows][b_cols] -> sum(A[a_rows][common_dim], B[common_dim][b_cols])
        // row major index = row * total_cols + col
        for i in 0..n{
            for c in 0..k{
                let a=self.data[(i*k)+c];
                for j in 0..m{
                    let b=other_tensor.data[(c*m)+j];
                    new_tensor_data[(i*m)+j]+=a*b;
                }
            }
        }
        Ok(Tensor { data:new_tensor_data, shape:vec![n,m]})
    }
    // matmut parallel
    pub fn matmul_parallel(&self, other: &Tensor) -> Result<Tensor, Error> {
        if self.shape[1] != other.shape[0] { 
            return Err(Error::ShapeMismatchError); 
        }
        let n = self.shape[0]; 
        let k = self.shape[1]; 
        let m = other.shape[1]; 
        let other_t = other.transpose();
        let mut new_tensor_data = vec![0.0; n*m];
        new_tensor_data.par_chunks_mut(m) 
            .enumerate() 
            .for_each(|(i, row_slice)| {
                for j in 0..m {
                    for c in 0..k {
                        let a = self.data[(i * k) + c];
                        let b = other_t.data[(j * k) + c];
                        row_slice[j]+=a*b;
                    }
                }
            });
        Ok(Tensor { data: new_tensor_data, shape: vec![n,m] })
    }

    // pub fn matmul_parallel_ikj(&self, other: &Tensor) -> Result<Tensor, Error> {
    //     if self.shape[1] != other.shape[0] { return Err(Error::ShapeMismatchError); }
    //     let n = self.shape[0]; 
    //     let k = self.shape[1]; 
    //     let m = other.shape[1]; 
    //     let mut new_data = vec![0.0; n * m];
    //     new_data.par_chunks_mut(m)
    //         .enumerate()
    //         .for_each(|(i, row_slice)| {
    //             for c in 0..k {
    //                 let a = self.data[(i * k) + c];
    //                 for j in 0..m { 
    //                     let b = other.data[(c * m) + j];
    //                     row_slice[j] += a * b;
    //                 }
    //             }
    //         });

    //     Ok(Tensor { data: new_data, shape: vec![n, m] })
    // }

    pub fn transpose(&self) -> Tensor {
        let mut new_tensor_data:Vec<f32> = vec![0.0;self.get_rows()*self.get_columns()];
        for i in 0..self.get_rows(){
            for j in 0..self.get_columns(){
                new_tensor_data[(j*self.get_rows())+i]=self.data[(i*self.get_columns())+j] 
            }
        }
        Tensor { data: new_tensor_data , shape:vec![self.get_columns(),self.get_rows()]}
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

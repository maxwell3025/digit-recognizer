pub struct NetworkParameters{
    pub layer12: nalgebra::DMatrix<f64>,
    pub layer23: nalgebra::DMatrix<f64>,
}

const LAYER_1_SIZE: usize = 784;
const LAYER_2_SIZE: usize = 16;
const LAYER_3_SIZE: usize = 10;

impl NetworkParameters{
    pub fn zeroes() -> Self{
        NetworkParameters {
            layer12: nalgebra::DMatrix::zeros(LAYER_2_SIZE, LAYER_1_SIZE + 1),
            layer23: nalgebra::DMatrix::zeros(LAYER_3_SIZE, LAYER_2_SIZE + 1),
        }
    }

    pub fn new() -> Self{
        NetworkParameters {
            layer12: nalgebra::DMatrix::from_fn(LAYER_2_SIZE, LAYER_1_SIZE + 1, |_, _| rand::random::<f64>() - 0.5),
            layer23: nalgebra::DMatrix::from_fn(LAYER_3_SIZE, LAYER_2_SIZE + 1, |_, _| rand::random::<f64>() - 0.5),
        }
    }

    pub fn gradient(&self, input: &crate::image::Image, label: u8) -> NetworkParameters{
        let layer_1 = nalgebra::DVector::from_vec(input.data.clone().into_iter().map(|x| x as f64/256.).collect()).insert_row(0, 1.0);
        let layer_2 = (self.layer12.clone() * layer_1.clone()).map(sigmoid).insert_row(0, 1.0);
        let layer_3 = (self.layer23.clone() * layer_2.clone()).map(sigmoid);
        let predictions = layer_3.clone() / layer_3.clone().sum();
        let mut prediction_grad = predictions.clone();
        prediction_grad[label as usize] -= 1.0;
        let sum_gradient = - layer_3.dot(&prediction_grad) / (layer_3.sum() * layer_3.sum());
        let layer_3_partial = prediction_grad / layer_3.sum();
        let layer_3_grad = layer_3_partial + nalgebra::DVector::from_element(LAYER_3_SIZE, sum_gradient);
        let layer_3_input_grad = layer_3_grad.component_mul(&layer_3.map(|x| x * (1.0 - x)));
        let layer_2_grad = self.layer23.transpose() * layer_3_input_grad.clone();
        let layer_2_input_grad = layer_2_grad.component_mul(&layer_2.map(|x| x * (1.0 - x))).remove_row(0);
        let layer_1_grad = self.layer12.transpose() * layer_2_input_grad.clone();
        NetworkParameters{
            layer12: layer_2_input_grad * layer_1.transpose(),
            layer23: layer_3_input_grad * layer_2.transpose(),
        }
    }

    pub fn loss(&self, input: &crate::image::Image, label: u8) -> f64{
        let layer_1 = nalgebra::DVector::from_vec(input.data.clone().into_iter().map(|x| x as f64/256.).collect()).insert_row(0, 1.0);
        let layer_2 = (self.layer12.clone() * layer_1.clone()).map(sigmoid).insert_row(0, 1.0);
        let layer_3 = (self.layer23.clone() * layer_2.clone()).map(sigmoid);
        let predictions = layer_3.clone() / layer_3.clone().sum();
        let mut truth = nalgebra::DVector::zeros(LAYER_3_SIZE);
        truth[label as usize] = 1.0;
        (predictions - truth).norm_squared() * 0.5
    }

    pub fn square(&self) -> NetworkParameters{
        NetworkParameters {
            layer12: self.layer12.map(|x| x * x),
            layer23: self.layer23.map(|x| x * x),
        }
    }

    pub fn adam_term(&self, epsilon: f64) -> NetworkParameters{
        NetworkParameters {
            layer12: self.layer12.map(|x| x.sqrt() + epsilon),
            layer23: self.layer23.map(|x| x.sqrt() + epsilon),
        }
    }

    pub fn divide(&self, rhs: &NetworkParameters) -> NetworkParameters{
        NetworkParameters {
            layer12: self.layer12.component_div(&rhs.layer12),
            layer23: self.layer23.component_div(&rhs.layer23)
        }
    }
}

impl std::clone::Clone for NetworkParameters{
    fn clone(&self) -> Self{
        NetworkParameters {
            layer12: self.layer12.clone(),
            layer23: self.layer23.clone(),
        }
    }
}

impl std::ops::Mul<f64> for NetworkParameters{
    type Output = NetworkParameters;
    
    fn mul(self, rhs: f64) -> Self{
        NetworkParameters{
            layer12: self.layer12 * rhs,
            layer23: self.layer23 * rhs,
        }
    }
}

impl std::ops::Add for NetworkParameters{
    type Output = NetworkParameters;

    fn add(self, rhs: Self) -> Self{
        NetworkParameters{
            layer12: self.layer12 + rhs.layer12,
            layer23: self.layer23 + rhs.layer23,
        }
    }
}

fn sigmoid(x: f64) -> f64{
    1.0 / ((-x).exp() + 1.0)
}

fn sigmoid_derivative(x: f64) -> f64{
    1.0 / (2.0 + x.exp() + (-x).exp())
}

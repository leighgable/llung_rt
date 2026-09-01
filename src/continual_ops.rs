use std::fmt;
use tract_onnx::prelude::*;

// internal state for custom operation
#[derive(Clone, Debug, Hash)]
pub struct ContinualWeightUpdateOp {
    pub learning_rate: f32,
}

// the core execution trait
impl Op for ContinualWeightUpdateOp {
    fn name(&self) -> std::borrow::Cow<str> {
        "ContinualWeightUpdate".into()
    }

    // allows deeper graph level rewrites/optimizations
    fn as_typed(&self) -> Option<&dyn TypedOp> {
        Some(self)
    }
}

// actual mathematical forwarding logic
impl EvalOp for ContinualWeightUpdateOp {
    fn is_stateless(&self) -> bool {
        false // continual learning alters internal state dynamically
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input_tensor = args_1!(inputs);

        // convert the incoming abstract tract-tensor into an ndarray view for quick math
        let view = input_tensor.to_array_view::<f32>()?;

        tracing::info!("Custom Op Intercepted Tensor Shape: {:?}", view.shape());

        // --- EXPERIMENTAL CONTINUAL LEARNING CODE GOES HERE ---
        // mutate heap-allocated weights based on inputs or
        // track dynamic gradients completely decoupled from default backprop.
        tracing::info!(
            "Simulating backprop fragment update loop at LR: {}",
            self.learning_rate
        );
        // -----------------------------------------------------

        // For this boilerplate example, we pass the data through unmodified (Identity)
        Ok(tvec!(input_tensor.clone()))
    }
}

// serialization and layout constraints for compiler type inference
impl TypedOp for ContinualWeightUpdateOp {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        // pass-through operation doesn't mutate tensor, outputs match inputs
        Ok(tvec!(inputs[0].clone()))
    }

    // operational code direct to the evaluation pipeline
    fn as_op(&self) -> &dyn Op {
        self
    }
    fn as_op_mut(&mut self) -> &mut dyn Op {
        self
    }
}

use std::path::Path;
use tract_onnx::prelude::*;

fn main() -> anyhow::Result<()> {
    let model_path = Path::new("model.onnx");

    if !model_path.exists() {
        println!("⚠️ No 'model.onnx' found. Please place a model at the root directory.");
        return Ok(());
    }

    println!("🧠 Loading model graph natively via Tract...");

    // 1. Load the model layout as a mutable structure
    // This allows you to append nodes, extract weights, or apply custom learning parameters
    let mut model = tract_onnx::onnx().model_for_path(model_path)?;

    // Example: Read properties of the graph for custom logic
    println!("Graph contains {} operation nodes.", model.nodes.len());

    // 2. Execution dispatch loop tailored for your laptops
    #[cfg(feature = "gpu")]
    {
        // Try executing via pure Rust Vulkan bindings (Works on Framework & ThinkPad iGPUs)
        println!("🚀 Dispatching execution plan to Vulkan iGPU via Wonnx...");
        // let session = wonnx::Session::from_path(model_path).block_on()?;
    }

    // Fallback: Ultra-fast embedded CPU layout execution built by Sonos
    println!("🐌 Optimizing execution path for laptop x86 CPU...");
    let runnable_plan = model.into_runnable()?;

    println!("✅ Runtime environment configured successfully.");
    Ok(())
}

// use tract_onnx::prelude::*;
// use std::path::Path;

// mod continual_ops;
// use continual_ops::ContinualWeightUpdateOp;

// fn main() -> anyhow::Result<()> {
//     let model_path = Path::new("model.onnx");

//     if !model_path.exists() {
//         println!("❌ 'model.onnx' not found. Run python script to export it first.");
//         return Ok(());
//     }

//     // 1. Initialize standard ONNX frame reader
//     let onnx_reader = tract_onnx::onnx();

//     // 2. Parse model structure into a completely mutable graph layout
//     let mut model = onnx_reader.model_for_path(model_path)?;

//     println!("Original Model Nodes count: {}", model.nodes.len());

//     // 3. Scan the graph to find our target placeholder node from PyTorch
//     let mut target_node_id = None;
//     for node in model.nodes() {
//         // Locate nodes derived from the 'continual_layer' namespace
//         if node.name.contains("continual_layer") {
//             println!("🎯 Found placeholder node ID: {}, Name: {}", node.id, node.name);
//             target_node_id = Some(node.id);
//             break;
//         }
//     }

//     // 4. Swap the placeholder node with your custom Rust operation
//     if let Some(id) = target_node_id {
//         let custom_node = ContinualWeightUpdateOp { learning_rate: 0.005 };

//         // Mutate the model graph internally
//         model.node_mut(id).op = Box::new(custom_node);
//         println!("⚡ Successfully patched in 'ContinualWeightUpdateOp' node.");
//     }

//     // 5. Freeze the graph layout and prepare the highly optimized execution plan
//     let runnable = model.into_runnable()?;

//     // 6. Execute a test forward pass with a dummy tensor input to trigger the custom node
//     let input_data = tract_ndarray::Array2::<f32>::zeros((1, 10)); // Match shape of export script
//     let input_tensor: Tensor = input_data.into();

//     println!("🏃 Running inference test loop...");
//     let outputs = runnable.run(tvec!(input_tensor.into()))?;

//     println!("🏁 Execution complete. Outputs generated: {}", outputs.len());
//     Ok(())
// }

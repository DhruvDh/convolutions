use ndarray::prelude::*;
use ndarray::Zip;
use itertools::iproduct;
use std::time::Instant;
use rayon::prelude::*;

const NUM_THREADS: usize = 32;

fn do_it(kernel_shape: (usize, usize), img_shape: (usize, usize)) -> f32 {

    let kernel = Array::from_elem(kernel_shape, 1.5f32);
    let img = Array::from_elem(img_shape, 1.32f32);

    let kernel_offset = kernel_shape.0 / 2;

    let x_range = kernel_offset..(img.shape()[0] - kernel_offset);
    let y_range = kernel_offset..(img.shape()[1] - kernel_offset);

    let coords: Vec<(usize, usize)> = iproduct!(x_range, y_range).collect();

    let mut output: Vec<f32> = Vec::new();

    let now = Instant::now();
    
    output.par_extend(coords.par_iter().map(|(x, y)| {
        Zip::from(img.slice( s![
            (x - kernel_offset)..=(x + kernel_offset),
            (y - kernel_offset)..=(y + kernel_offset)
            ] )).and(&kernel)
                .fold(0f32, |acc, i, k| acc + (i * k))

    }));

    let output = Array::from_shape_vec(
        (img.shape()[0] - (kernel_offset * 2), img.shape()[1] - (kernel_offset * 2)),
         output
    ).unwrap();

    let time_taken = now.elapsed().as_secs_f32();

    dbg!(&output.sum());

    let pixels  = output.shape()[0] * output.shape()[1];
    
    (pixels as f32 * 10e-9) / time_taken
}

fn main() {
    rayon::ThreadPoolBuilder::new().num_threads(NUM_THREADS).build_global().unwrap();

    let kernels = vec![(3,3), (5,5), (9,9), (11,11), (13,13), (15,15)];
    let imgs = vec![(1024,768), (2048,2048), (8192,8192), (4194304,768), (16777216,768)];

    for (i, k) in iproduct!(imgs, kernels) {
        println!("{}x{}\tconvolution of\t{}x{}\t image proccessed at\t{} gigapixels/sec.", k.0, k.1, i.0, i.1, do_it(k, i));
    }
}

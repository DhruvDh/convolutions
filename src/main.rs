use itertools::iproduct;
use ndarray::prelude::*;
use ndarray::Zip;
use rayon::prelude::*;
use std::time::Instant;
use std::sync::Arc;
use parking_lot::{Mutex};

const NUM_THREADS: usize = 16;

fn do_it(kernel_shape: (usize, usize), img_shape: (usize, usize)) -> f32 {
    let mut img = Array::from_elem(img_shape, 1.32f32);

    let mut output = Arc::new(Mutex::new(Vec::new()));
    let kernel_offset = kernel_shape.0 / 2;
    
    let now = Instant::now();

    rayon::scope(|s| {
        let y_range: Vec<usize> = (0..img_shape.1).collect();
        let y_range = y_range.as_slice();

        // for patch in img.windows(kernel_shape).into_par_iter() {
        //     println!("{:?}", patch);
        //     break;
        // }

        for y_window in y_range.windows(kernel_shape.1) {
            let slab = img.slice(s![.., y_window[0]..*y_window.last().unwrap() + 1]);
            let kernel = Array::from_elem(kernel_shape.0 * kernel_shape.1, 1.5f32);
            let out = output.clone();

            s.spawn(move |s| {
                let x_range: Vec<usize> = (0..img_shape.0).collect();
                let x_range = x_range.as_slice();

                let mut patch_matrix = Array::from_elem(
                    (
                        img_shape.0 - kernel_shape.0 + 1,
                        kernel_shape.0 * kernel_shape.1,
                    ),
                    0f32,
                );

                // println!("{:?}", patch_matrix);
                // println!("{:?}", x_range.windows(kernel_shape.0).last());
                for x_window in x_range.windows(kernel_shape.0) {
                    patch_matrix
                        .row_mut(x_window[0])
                        .assign(&
                            slab.slice(s![x_window[0]..*x_window.last().unwrap() + 1, ..])
                                .to_owned()
                                .into_shape(kernel_shape.0 * kernel_shape.1)
                                .unwrap()
                        );
                }
        
                let mut _output = out.lock();
                (*_output).append(&mut kernel.dot(&(patch_matrix.t())).to_vec() );

            });
        }
        // output.par_extend(coords.into_par_iter().map(|(x, y)| {
        //     Zip::from(img.slice( s![
        //         (x - kernel_offset)..=(x + kernel_offset),
        //         (y - kernel_offset)..=(y + kernel_offset)
        //         ] )).and(&kernel)
        //             .fold(0f32, |acc, i, k| acc + (i * k))
        // }));

        // println!("{:?}", output.len());
    });

    
    let mut gaurd = output.lock();
    let out_vec: Vec<f32> = (*gaurd).drain(0..).collect();

    let output = Array::from_shape_vec(
        (img_shape.0 - (kernel_offset * 2), img_shape.1 - (kernel_offset * 2)),
        out_vec
    ).unwrap();
        
    let time_taken = now.elapsed().as_secs_f32();
    
    let pixels  = (img_shape.0 - kernel_shape.0 + 1) * (img_shape.1 - kernel_shape.1 + 1);
    (pixels as f32 * 10e-9) / time_taken
}

fn main() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(NUM_THREADS)
        .build_global()
        .unwrap();

    // let kernels = vec![(3, 3),]; 
    let kernels = vec![(3, 3), (5, 5), (9, 9), (11, 11), (13, 13), (15, 15)];
    let imgs = vec![
        (1024, 768),
        (2048, 2048),
        (8192, 8192),
        (4194304, 768),
        (16777216, 768),
    ];

    for (i, k) in iproduct!(imgs, kernels) {
        println!(
            "{}x{}\tconvolution of\t{}x{}\t image proccessed at\t{} gigapixels/sec.",
            k.0,
            k.1,
            i.0,
            i.1,
            do_it(k, i)
        );
    }
}

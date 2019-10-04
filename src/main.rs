use paste::{item};
use itertools::iproduct;

macro_rules! create_convolution_type {
    ($($x: expr),+) => {
        $(
            item! { type [<Conv $x>] = [[f32; $x]; $x]; }

            item! {
                fn [<new_ conv $x>]() -> [<Conv $x>] {
                    [[1.32f32; $x]; $x]
                }
            }
        )+
    };
}

macro_rules! create_output_type {
    ($I: expr, $X: expr, $Y: expr, [$($O: expr),+]) => {
        $(
            item! { type [<Out $I x $O>] = [[f32; $X - $O + 1]; $Y - $O + 1]; }

            item! {
                fn [<new_ out $I x $O>]() -> [<Out $I x $O>] {
                    [[0f32; $X - $O + 1]; $Y - $O + 1]
                }
            }
        )+
    };
}

macro_rules! create_image_type {
    ($index: expr,$x: expr, $y: expr) => {
        item! { type [<Img $index>] = [[f32; $x]; $y]; }

        item! {
            fn [<new_ img $index>]() -> [<Img $index>] {
                [[2.5f32; $x]; $y]
            }
        }

        create_output_type!($index, $x, $y, [3, 5, 7, 9, 11, 13, 15]);
    };
}

create_convolution_type!(3, 5, 7, 9, 11, 13, 15);
create_image_type!(1, 1024, 768);
create_image_type!(2, 2048, 2048);
create_image_type!(3, 8192, 8192);
create_image_type!(4, 4194304, 768);
create_image_type!(5, 16777216, 768);

fn main() {
    let img = new_img1();
    let kernel = new_conv9();

    let mut out = new_out1x9();

    let coords = iproduct!(0..out.len(), 0..out[0].len()); 

    let img_patches = iproduct!(0..out.len(), 0..out[0].len()).map(|(row_i, col_j)| {
        img[row_i..row_i+9].iter().map(move |s| &s[col_j..col_j+9]).flatten()
    });

    for patch in img_patches {
        // for i in patch {
        //     println!("{:?}", i);
        // }
        println!("{:?}", patch.count());
        break;
    }


    // coords.par_bridge().for_each(|(row_i, col_j)| {
    //     let img_patch = img[row_i..row_i+9].iter().map(|s| &s[col_j..col_j+9]);

    //     out[row_i][col_j] = img_patch.enumerate()
    //                             .map(|(row_x, val_x)| {
    //                                 val_x.iter().enumerate().map(move |(col_y, val_y)| {
    //                                     kernel[row_x][col_y] * val_y
    //                                 })
    //                             }).flatten()
    //                               .sum();
            
    //         dbg!(&out[row_i][col_j]);
    // });
}
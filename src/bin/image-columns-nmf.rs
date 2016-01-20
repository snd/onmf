use std::path::Path;

#[macro_use] extern crate clap;

extern crate image;

extern crate onmf;
use onmf::helpers::Array2D;

// `name` and `long` have the same lifetime
fn named_usize_arg<'n, 'h, 'g, 'p, 'r>(name: &'n str, help: &'h str) -> clap::Arg<'n, 'n, 'h, 'g, 'p, 'r> {
    // we don't own `name` but we now own `error_message`
    // which contains `name`.
    let error_message = format!(
        "value provided for argument `{}` is not a valid unsigned integer number",
        name);
    clap::Arg::with_name(name)
         .long(name)
         .help(help)
         // `move` makes it such that the  closure owns `error_message`
         .validator(move |val| {
             match val.parse::<usize>() {
                 Ok(_) => Ok(()),
                 // the closure owns `error_message`.
                 // we can't move `error_message` out of the closure.
                 // we have to clone it instead.
                 Err(_) => Err(error_message.clone())
             }
         })
         .takes_value(true)
}

fn main() {
    let default_nhidden: usize = 10;
    let help_nhidden = format!("number of hidden (latent) variables to find (defaults to `{}`)", default_nhidden);

    let help_input_image_path = "path to image on the the grayscale amounts in the columns of which the NMF should be applied";

    let matches =
        clap::App::new("image-columns-nmf")
            .version(&crate_version!()[..])
            .about("applies NMF on the grayscale amounts of the columns of an image")
            .arg(clap::Arg::with_name("input-image-path")
                 .index(1)
                 .help(&help_input_image_path)
                 .required(true))
            .arg(named_usize_arg("nhidden", &help_nhidden))
            .get_matches();

    let nhidden: usize = value_t!(matches.value_of("nhidden"), usize).unwrap_or(default_nhidden);
    let input_image_path_string = matches.value_of("input-image-path").unwrap();
    let input_image_path = Path::new(input_image_path_string);

    println!("nhidden = {}", nhidden);
    println!("input_image_path = {:?}", input_image_path);
    println!("input_image_path.parent() = {:?}", input_image_path.parent());
    println!("input_image_path.file_name() = {:?}", input_image_path.file_name());
    println!("input_image_path.file_stem() = {:?}", input_image_path.file_stem());
    println!("input_image_path.extension() = {:?}", input_image_path.extension());

    // let mut input_image_file = std::fs::File::open(&input_image_path).unwrap();

    let image = image::open(&input_image_path).unwrap();

    // to grayscale
    let image_luma = image.to_luma();

    println!("image_luma.dimensions() = {:?}", image_luma.dimensions());
    let (width, height) = image_luma.dimensions();

    let nobserved = height as usize;
    let nsamples = width as usize;
    println!("nobserved = {}", nobserved);
    println!("nsamples = {}", nsamples);

    // the indexing of image_array is reversed in respect to image_luma
    let mut samples = Array2D::<f32>::zeros((nsamples, nobserved));

    for (x, y, pixel) in image_luma.enumerate_pixels() {
        let index = (x as usize, y as usize);

        // convert from 0-255 to 0.-1.
        samples[index] = (pixel[0] as f32) / 255.;
    }

    // TODO 
    // let mut ortho_nmf = onmf::OrthogonalNMFBlas::new_random01(
    //     nhidden, nobserved, nsamples);

    // TODO iterate nmf
}

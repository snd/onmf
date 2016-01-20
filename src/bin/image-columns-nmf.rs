#[macro_use] extern crate clap;

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
    let input_image_path = matches.value_of("input-image-path").unwrap();

    println!("nhidden = {}", nhidden);
    println!("input_image_path = {}", input_image_path);
}

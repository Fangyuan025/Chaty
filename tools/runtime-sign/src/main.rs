//! Sign Chaty's runtime pack index — the app takes `runtimes.json` only with a
//! `runtimes.json.sig` made by the key whose public half it carries
//! (`INDEX_PUBKEY` in src-tauri/src/runtimes.rs).
//!
//!   runtime-sign keygen <private-key-file>   writes the private key (PKCS#8,
//!                                            base64) to the file, prints the
//!                                            public key (raw Ed25519, base64)
//!   runtime-sign sign <file>                 signs with RUNTIMES_SIGNING_KEY
//!                                            (the private key), writes <file>.sig
//!   runtime-sign verify <file> <public-key>  checks <file>.sig
use base64::Engine as _;
use ring::signature::{Ed25519KeyPair, KeyPair, UnparsedPublicKey, ED25519};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let b64 = base64::engine::general_purpose::STANDARD;
    let fail = |msg: &str| -> ! {
        eprintln!("runtime-sign: {msg}");
        std::process::exit(1)
    };
    match args.get(1).map(String::as_str) {
        Some("keygen") => {
            let out = args.get(2).unwrap_or_else(|| fail("keygen <private-key-file>"));
            let rng = ring::rand::SystemRandom::new();
            let pkcs8 = Ed25519KeyPair::generate_pkcs8(&rng).unwrap_or_else(|_| fail("could not make a key"));
            let pair = Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap_or_else(|_| fail("bad key"));
            std::fs::write(out, b64.encode(pkcs8.as_ref())).unwrap_or_else(|e| fail(&e.to_string()));
            println!("{}", b64.encode(pair.public_key().as_ref()));
        }
        Some("sign") => {
            let file = args.get(2).unwrap_or_else(|| fail("sign <file>"));
            let key = std::env::var("RUNTIMES_SIGNING_KEY").unwrap_or_else(|_| fail("RUNTIMES_SIGNING_KEY is not set"));
            let pkcs8 = b64.decode(key.trim()).unwrap_or_else(|_| fail("RUNTIMES_SIGNING_KEY is not base64"));
            let pair = Ed25519KeyPair::from_pkcs8(&pkcs8).unwrap_or_else(|_| fail("RUNTIMES_SIGNING_KEY is not a key"));
            let body = std::fs::read(file).unwrap_or_else(|e| fail(&e.to_string()));
            std::fs::write(format!("{file}.sig"), b64.encode(pair.sign(&body).as_ref())).unwrap_or_else(|e| fail(&e.to_string()));
        }
        Some("verify") => {
            let (file, public) = match (args.get(2), args.get(3)) {
                (Some(f), Some(p)) => (f, p),
                _ => fail("verify <file> <public-key>"),
            };
            let body = std::fs::read(file).unwrap_or_else(|e| fail(&e.to_string()));
            let sig = std::fs::read_to_string(format!("{file}.sig")).unwrap_or_else(|e| fail(&e.to_string()));
            let key = b64.decode(public.trim()).unwrap_or_else(|_| fail("public key is not base64"));
            let sig = b64.decode(sig.trim()).unwrap_or_else(|_| fail("signature is not base64"));
            if UnparsedPublicKey::new(&ED25519, key).verify(&body, &sig).is_err() {
                fail("signature does not match");
            }
            println!("ok");
        }
        _ => fail("keygen <private-key-file> | sign <file> | verify <file> <public-key>"),
    }
}

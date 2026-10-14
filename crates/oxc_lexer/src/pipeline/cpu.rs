//! Types representing CPU feature sets, used to choose between SIMD and generic implementations.
//!
//! - `Avx2`: x86_64 with AVX2, BMI2 and POPCNT.
//! - `Neon`: aarch64 with Neon.
//! - [`Generic`]: No SIMD. Exists on all platforms.
//!
//! [`DynCpu`] is an enum of these types. [`DynCpu::detect`] returns the best one available.
//!
//! Code which has SIMD implementations takes a type implementing [`Cpu`],
//! and uses [`cpu_select!`] to pick an implementation.
//! [`cpu_versions!`] turns a function into one with a separate entry point for each CPU feature set.
//!
//! * Given a specific type (e.g. `Avx2`): the choice is made at compile time.
//! * Given a [`DynCpu`]: the choice is made at runtime.
//!
//! Functions which use [`cpu_select!`] typically take a generic `impl Cpu`.
//! This compiles a separate version of the function for each [`Cpu`] type that callers use with it.
//!
//! [`DynCpu`] also implements [`Cpu`], so functions that receive an `impl Cpu`
//! can do either a compile-time or runtime features check, depending on whether it's called with
//! a concrete type (e.g. `Avx2`) or [`DynCpu`].
//!
//! `Avx2`, `Neon`, and [`Generic`] types are all zero-sized, so can be passed between functions
//! without any runtime cost. They operate at the level of the type system only.
//!
//! # SAFETY
//!
//! These types are unforgeable.
//!
//! * Code outside this file can't create them.
//! * Each has a private field, and `DynCpu::detect` is the only code which creates them.
//!   It only creates an `Avx2` or a `Neon` if that feature set is available.
//! * A [`DynCpu`] can only be built from one of these types, so it's unforgeable too.
//! * [`Cpu`] trait is sealed, so no other type can implement it.
//!
//! Therefore, holding one of these types guarantees that that the CPU supports that feature set.
//!
//! The rest of the lexer relies on this for safety. A function which takes `Avx2` or `Neon`
//! can use instructions that rely on support for those features without any checks.
//!
//! DO NOT add any other way to create these types (e.g. a `Default` impl or a public constructor).

/// Trait for types representing a CPU feature set.
///
/// Implemented by all types representing specific CPU feature sets, and by [`DynCpu`].
#[expect(private_bounds)]
pub(super) trait Cpu: Sealed + Copy {
    fn to_dyn(self) -> DynCpu;
}

/// Private trait to guarantee that only types defined in this file can implement [`Cpu`].
trait Sealed {}

/// Enum representing a CPU feature set.
///
/// Used for runtime feature detection.
///
/// Implements [`Cpu`] and also provides a [`detect`] method which initializes [`DynCpu`]
/// via target feature detection.
///
/// [`detect`]: Self::detect
#[derive(Clone, Copy, Debug)]
pub(super) enum DynCpu {
    /// AVX2 feature set. Only available on x86_64.
    #[cfg(target_arch = "x86_64")]
    Avx2(Avx2),
    /// Neon feature set. Only available on aarch64.
    #[cfg(target_arch = "aarch64")]
    Neon(Neon),
    /// Exists on all platforms, but can never be created on aarch64
    /// when Neon is enabled at compile time. See [`Generic`].
    Generic(Generic),
}

impl Cpu for DynCpu {
    /// Get [`DynCpu`].
    ///
    /// Just returns itself.
    #[inline(always)]
    fn to_dyn(self) -> DynCpu {
        self
    }
}

impl Sealed for DynCpu {}

/// Generic implementation.
mod generic {
    use super::*;

    /// Generic CPU feature set.
    ///
    /// On aarch64 with Neon enabled at compile time, this type is uninhabited - `Generic` can never be created.
    /// So [`DynCpu::Generic`] can never exist there, and the compiler removes code which handles it.
    /// Every aarch64 target with `std` enables Neon by default, so in practice this applies to all aarch64 builds.
    ///
    /// The type still exists there, so code which uses it is still compiled and type-checked.
    /// That means generic code is still linted, and is visible to rust-analyzer, when working on an ARM Mac.
    #[derive(Clone, Copy, Debug)]
    pub struct Generic(GenericInner);

    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    type GenericInner = ();

    // When Neon is enabled at compile time, Neon is always available, so the generic implementation
    // is never needed. `Infallible` has no values, which makes `Generic` impossible to create.
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    type GenericInner = std::convert::Infallible;

    impl Cpu for Generic {
        /// Get [`DynCpu`] representing generic CPU feature set.
        #[inline(always)]
        fn to_dyn(self) -> DynCpu {
            DynCpu::Generic(self)
        }
    }

    impl Sealed for Generic {}

    impl DynCpu {
        /// Create [`DynCpu::Generic`].
        ///
        /// Private within this file. Used only in [`detect`] implementations.
        /// Not available on aarch64 with Neon enabled at compile time, where a `Generic` can't be created.
        ///
        /// [`detect`]: Self::detect
        #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
        pub(super) fn generic() -> Self {
            Generic(()).to_dyn()
        }
    }
}

pub(super) use generic::Generic;

/// x86_64 features.
/// AVX2 or generic.
#[cfg(target_arch = "x86_64")]
mod x86_64 {
    use super::*;

    /// x64_64 AVX2 CPU feature set.
    #[derive(Clone, Copy, Debug)]
    pub struct Avx2(());

    impl Cpu for Avx2 {
        /// Get [`DynCpu`] representing AVX2 CPU feature set.
        #[inline(always)]
        fn to_dyn(self) -> DynCpu {
            DynCpu::Avx2(self)
        }
    }

    impl Sealed for Avx2 {}

    impl DynCpu {
        /// Detect CPU feature set for x86_64 platform.
        pub fn detect() -> Self {
            // Currently a compile-time check. Later on will be replaced by runtime detection.
            if cfg!(all(
                target_feature = "avx2",
                target_feature = "bmi2",
                target_feature = "popcnt"
            )) {
                Avx2(()).to_dyn()
            } else {
                DynCpu::generic()
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub(super) use x86_64::Avx2;

/// aarch64 features.
/// Neon or generic.
#[cfg(target_arch = "aarch64")]
mod aarch64 {
    use super::*;

    /// aarch64 Neon CPU feature set.
    #[derive(Clone, Copy, Debug)]
    pub struct Neon(());

    impl Cpu for Neon {
        /// Get [`DynCpu`] representing Neon CPU feature set.
        #[inline(always)]
        fn to_dyn(self) -> DynCpu {
            DynCpu::Neon(self)
        }
    }

    impl Sealed for Neon {}

    impl DynCpu {
        /// Detect CPU feature set for aarch64 platform.
        pub fn detect() -> Self {
            // Neon is always enabled except on `aarch64*-unknown-none-softfloat` platforms
            // or when it's specifically disabled by compiling with `-C target-feature=-neon`.
            // Either way, it can be detected at compile time. No runtime feature check.
            cfg_select! {
                target_feature = "neon" => Neon(()).to_dyn(),
                _ => DynCpu::generic(),
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub(super) use aarch64::Neon;

/// Fallback implementation for other platforms.
///
/// Only generic feature set is available.
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
impl DynCpu {
    /// Detect CPU feature set for other platform.
    ///
    /// Only generic feature set is available.
    pub fn detect() -> Self {
        DynCpu::generic()
    }
}

/// Macro for either static or dynamic feature selection.
///
/// Arms use normal `match` syntax, but there must be exactly these 3 arms, in this order:
///
/// - `Avx2(...)`: Only compiled on x86_64.
/// - `Neon(...)`: Only compiled on aarch64.
/// - `Generic(...)`: Compiled everywhere.
///   On aarch64 with Neon enabled at compile time, it's compiled and type-checked,
///   but can never run, so the compiler removes it. See [`Generic`].
///
/// Each arm binds its CPU feature set type (e.g. `Avx2`), so it can be passed on.
/// Fallback arms (e.g. `_ => ...`) and match guards are not allowed.
///
/// Requiring all 3 arms means every call site says what to do on every platform.
/// It also makes the `match` exhaustive on every platform, without a fallback arm.
///
/// # Examples
///
/// ## Static selection
///
/// ```ignore
/// use crate::pipeline::cpu::{Cpu, cpu_select};
///
/// fn classify(&mut self, cpu: impl Cpu) {
///     // Compiles down to a single arm (no runtime branch)
///     cpu_select!(cpu, {
///         Avx2(cpu) => classify_avx2(self, cpu),
///         Neon(cpu) => classify_neon(self, cpu),
///         Generic(_) => classify_generic(self),
///     });
/// }
/// ```
///
/// ## Dynamic selection
///
/// ```ignore
/// use crate::pipeline::cpu::{DynCpu, cpu_select};
///
/// fn classify(cpu: DynCpu) {
///     // Runtime match over options available on the current platform
///     cpu_select!(cpu, {
///         Avx2(cpu) => classify_avx2(self, cpu),
///         Neon(cpu) => classify_neon(self, cpu),
///         Generic(_) => classify_generic(self),
///     });
/// }
/// ```
///
/// ## Expansion
///
/// Macro expands to:
///
/// ```ignore
/// match Cpu::to_dyn(cpu) {
///     #[cfg(target_arch = "x86_64")]
///     DynCpu::Avx2(cpu) => classify_avx2(self, cpu),
///
///     #[cfg(target_arch = "aarch64")]
///     DynCpu::Neon(cpu) => classify_neon(self, cpu),
///
///     DynCpu::Generic(_) => classify_generic(self),
/// }
/// ```
macro_rules! cpu_select {
    // Tokens of the arms are consumed one at a time, and copied into `[...]`.
    // `Avx2(...) =>` is rewritten to `DynCpu::Avx2(...) =>`, with a `#[cfg]` attribute for x86_64.
    // `Neon(...) =>` is rewritten the same way, with a `#[cfg]` attribute for aarch64.
    // `Generic(...) =>` is rewritten to `DynCpu::Generic(...) =>`, with no `#[cfg]` attribute.
    // All other tokens are copied unchanged, so the arm bodies are parsed by the `match` itself.
    //
    // `[no no no]` records whether the `Avx2`, `Neon` and `Generic` arms have been seen yet.
    // Each arm's rule only matches if the arms before it have been seen, and it hasn't.
    // So the arms must come in that order, once each. Anything else is a compile error.

    // Entry point
    ($cpu:expr, { $($arms:tt)* }) => {
        crate::pipeline::cpu::cpu_select!(@munch $cpu; [no no no]; []; $($arms)*)
    };

    // `Avx2` arm. Must be first.
    (
        @munch $cpu:expr; [no no no]; [$($done:tt)*];
        Avx2($($pat:tt)*) => $($rest:tt)*
    ) => {
        crate::pipeline::cpu::cpu_select!(
            @munch $cpu;
            [yes no no];
            [
                $($done)*
                #[cfg(target_arch = "x86_64")]
                crate::pipeline::cpu::DynCpu::Avx2($($pat)*) =>
            ];
            $($rest)*
        )
    };

    // `Neon` arm. Must follow `Avx2` arm.
    (
        @munch $cpu:expr; [yes no no]; [$($done:tt)*];
        Neon($($pat:tt)*) => $($rest:tt)*
    ) => {
        crate::pipeline::cpu::cpu_select!(
            @munch $cpu;
            [yes yes no];
            [
                $($done)*
                #[cfg(target_arch = "aarch64")]
                crate::pipeline::cpu::DynCpu::Neon($($pat)*) =>
            ];
            $($rest)*
        )
    };

    // `Generic` arm. Must follow `Neon` arm.
    (
        @munch $cpu:expr; [yes yes no]; [$($done:tt)*];
        Generic($($pat:tt)*) => $($rest:tt)*
    ) => {
        crate::pipeline::cpu::cpu_select!(
            @munch $cpu;
            [yes yes yes];
            [
                $($done)*
                crate::pipeline::cpu::DynCpu::Generic($($pat)*) =>
            ];
            $($rest)*
        )
    };

    // Any other `Name(...) =>` arm - out of order, repeated, or not one of the 3 names
    (@munch $cpu:expr; $seen:tt; $done:tt; $name:ident($($pat:tt)*) => $($rest:tt)*) => {
        crate::pipeline::cpu::cpu_select!(@error)
    };

    // Any other `=>` - an arm whose pattern isn't `Name(...)` (e.g. `_ =>`), or which has a match guard
    (@munch $cpu:expr; $seen:tt; $done:tt; => $($rest:tt)*) => {
        crate::pipeline::cpu::cpu_select!(@error)
    };

    // Any other token.
    //
    // Copy 1 token, and go round again.
    //
    // The arm rules only consume `Avx2(...) =>` etc, not the arm's body.
    // So the remaining tokens can't all be copied in one go. They start with the body,
    // and a `Neon(...) =>` which still needs rewriting may come after it.
    //
    // `macro_rules!` can't match "all tokens up to the next `Neon`",
    // because `$($x:tt)*` would also match `Neon`.
    // The only way to find the next arm is to step through the tokens.
    //
    // Consuming the body in the arm rules would avoid this, but would need 2 rules for each arm
    // (block body, or expression followed by `,`), and would reject comma-free bodies like `unsafe { ... }`.
    //
    // A delimited group (e.g. a block body) is a single token, so its contents are never inspected.
    // This rule must come after the arm rules and the error rules above,
    // because `$next:tt` also matches `Avx2` and `=>`.
    (@munch $cpu:expr; $seen:tt; [$($done:tt)*]; $next:tt $($rest:tt)*) => {
        crate::pipeline::cpu::cpu_select!(
            @munch $cpu;
            $seen;
            [$($done)* $next];
            $($rest)*
        )
    };

    // All tokens consumed, and all 3 arms seen
    (@munch $cpu:expr; [yes yes yes]; [$($arms:tt)*];) => {
        match crate::pipeline::cpu::Cpu::to_dyn($cpu) {
            $($arms)*
        }
    };

    // All tokens consumed, but an arm is missing
    (@munch $cpu:expr; $seen:tt; $done:tt;) => {
        crate::pipeline::cpu::cpu_select!(@error)
    };

    // Error for any arms which don't follow the rules
    (@error) => {
        compile_error!(
            "`cpu_select!` must have arms `Avx2(...)`, `Neon(...)` and `Generic(...)`, in that order, once each"
        )
    };
}

pub(super) use cpu_select;

/// Macro to turn a function into one with a separate entry point for each CPU feature set.
///
/// Used for pipeline stages. Each entry point is compiled with its feature set's target features enabled,
/// so SIMD functions which the stage calls can be inlined into it.
///
/// Applied to a function with `#[apply(cpu_versions)]`:
///
/// ```ignore
/// use crate::pipeline::cpu::{Cpu, apply, cpu_versions};
///
/// /// Compress tokens.
/// #[apply(cpu_versions)]
/// #[inline(never)]
/// pub unsafe fn compress(src: &[u8], n: usize, cpu: impl Cpu) -> usize {
///     // ...
/// }
/// ```
///
/// `compress` becomes a function which calls one of 3 entry points, depending on the type of `cpu`.
/// It's `#[inline(always)]`, so when called with a specific type (e.g. `Avx2`),
/// it compiles to a direct call to that entry point.
///
/// The entry points are functions defined inside `compress`:
///
/// - `__avx2`: Takes `cpu: Avx2`. Compiled with AVX2 enabled. Only exists on x86_64.
/// - `__neon`: Takes `cpu: Neon`. Compiled with Neon enabled. Only exists on aarch64.
/// - `__generic`: Takes `cpu: Generic`.
///
/// Each entry point calls `__impl`, an `#[inline(always)]` function containing the original function body.
/// So the body is compiled as part of each entry point, and the AVX2 functions it calls can be inlined
/// into the AVX2 entry point. They can't be inlined into a function which doesn't have AVX2 enabled.
///
/// A function defined inside `compress` shadows a parameter with the same name in its body.
/// The names start with `__`, so they don't clash with any real parameter names.
///
/// Code for other platforms is removed with `cfg_select!`, not `#[cfg]` attributes.
/// That code contains copies of the function's parameters.
/// Rust Analyzer greys out the original code of anything in a node disabled by `#[cfg]`,
/// so `#[cfg]` (or [`cpu_select!`], which uses it) would grey out the function's signature.
/// Branches of `cfg_select!` are just dropped, so nothing is greyed out.
///
/// `#[inline]` and `#[inline(never)]` attributes go on the entry points.
/// All other attributes (e.g. doc comments) go on `compress`.
///
/// `#[inline(never)]` keeps the entry points as separate functions.
/// Without it, they can be inlined into their callers in builds which enable AVX2 globally.
/// `#[inline(always)]` isn't allowed, because it can't be combined with `#[target_feature]`.
///
/// The last parameter must be `cpu: impl Cpu` (any name instead of `cpu` is fine).
/// The other parameters must be `name: Type` (no patterns), and can't be called `__impl`, `__avx2`, etc.
///
/// ## Expansion
///
/// The example above expands to:
///
/// ```ignore
/// /// Compress tokens.
/// #[inline(always)]
/// pub unsafe fn compress(src: &[u8], n: usize, cpu: impl Cpu) -> usize {
///     #[inline(always)]
///     unsafe fn __impl(src: &[u8], n: usize, cpu: impl Cpu) -> usize {
///         // ...
///     }
///
///     #[inline(never)]
///     unsafe fn __generic(src: &[u8], n: usize, cpu: Generic) -> usize {
///         __impl(src, n, cpu)
///     }
///
///     cfg_select! {
///         target_arch = "x86_64" => {
///             #[inline(never)]
///             #[target_feature(enable = "avx2,bmi2,popcnt")]
///             unsafe fn __avx2(src: &[u8], n: usize, cpu: Avx2) -> usize {
///                 __impl(src, n, cpu)
///             }
///
///             match Cpu::to_dyn(cpu) {
///                 // SAFETY: `Avx2` type guarantees the CPU supports the features `__avx2` uses
///                 DynCpu::Avx2(cpu) => unsafe { __avx2(src, n, cpu) },
///                 DynCpu::Generic(cpu) => __generic(src, n, cpu),
///             }
///         }
///         target_arch = "aarch64" => {
///             #[inline(never)]
///             #[target_feature(enable = "neon")]
///             unsafe fn __neon(src: &[u8], n: usize, cpu: Neon) -> usize {
///                 __impl(src, n, cpu)
///             }
///
///             match Cpu::to_dyn(cpu) {
///                 // SAFETY: `Neon` type guarantees the CPU supports the features `__neon` uses
///                 DynCpu::Neon(cpu) => unsafe { __neon(src, n, cpu) },
///                 DynCpu::Generic(cpu) => __generic(src, n, cpu),
///             }
///         }
///         _ => {
///             match Cpu::to_dyn(cpu) {
///                 DynCpu::Generic(cpu) => __generic(src, n, cpu),
///             }
///         }
///     }
/// }
/// ```
#[expect(unused_macros)]
macro_rules! cpu_versions {
    // `#[inline]` attributes go on the entry points, and all other attributes go on the outer function.
    // Attributes are captured as raw tokens, so `inline` can be matched. They're sorted one at a time
    // into 2 lists `[...]`.
    //
    // `[unsafe]` or `[]` records whether the function is `unsafe`, and is copied onto every function.
    //
    // The last parameter gets a different type in each function, so it has to be matched on its own.
    // `macro_rules!` can't match "all parameters except the last one" in one go.
    // So parameters are moved into `[...]` one at a time, until only the last one is left.

    // Entry point: `unsafe fn`
    (
        $(#[$($attr:tt)*])*
        $vis:vis unsafe fn $name:ident($($params:tt)*) $(-> $ret:ty)? $body:block
    ) => {
        crate::pipeline::cpu::cpu_versions!(
            @attrs [] []; [$([$($attr)*])*]; [$vis [unsafe] $name ($($ret)?) $body]; $($params)*
        );
    };

    // Entry point: safe `fn`
    (
        $(#[$($attr:tt)*])*
        $vis:vis fn $name:ident($($params:tt)*) $(-> $ret:ty)? $body:block
    ) => {
        crate::pipeline::cpu::cpu_versions!(
            @attrs [] []; [$([$($attr)*])*]; [$vis [] $name ($($ret)?) $body]; $($params)*
        );
    };

    // `#[inline]` or `#[inline(...)]` attribute: Goes on the entry points
    (
        @attrs [$($outer:tt)*] [$($entry:tt)*];
        [[inline $($attr:tt)*] $($attrs:tt)*];
        $($rest:tt)*
    ) => {
        crate::pipeline::cpu::cpu_versions!(
            @attrs [$($outer)*] [$($entry)* #[inline $($attr)*]]; [$($attrs)*]; $($rest)*
        );
    };

    // Any other attribute: Goes on the outer function
    (
        @attrs [$($outer:tt)*] [$($entry:tt)*];
        [[$($attr:tt)*] $($attrs:tt)*];
        $($rest:tt)*
    ) => {
        crate::pipeline::cpu::cpu_versions!(
            @attrs [$($outer)* #[$($attr)*]] [$($entry)*]; [$($attrs)*]; $($rest)*
        );
    };

    // All attributes sorted
    (@attrs $outer:tt $entry:tt; []; [$($header:tt)*]; $($params:tt)*) => {
        crate::pipeline::cpu::cpu_versions!(
            @params [$outer $entry $($header)*]; []; $($params)*
        );
    };

    // Last parameter: `cpu: impl Cpu`
    (@params $header:tt; [$($arg:ident: $ty:ty,)*]; $cpu:ident: impl $cpu_trait:ident $(,)?) => {
        crate::pipeline::cpu::cpu_versions!(
            @expand $header; [$($arg: $ty,)*]; $cpu: impl $cpu_trait
        );
    };

    // Any other parameter
    (@params $header:tt; [$($done:tt)*]; $arg:ident: $ty:ty, $($rest:tt)*) => {
        crate::pipeline::cpu::cpu_versions!(
            @params $header; [$($done)* $arg: $ty,]; $($rest)*
        );
    };

    // Last parameter isn't `cpu: impl Cpu`, or a parameter isn't `name: Type`
    (@params $($rest:tt)*) => {
        compile_error!(
            "`cpu_versions!` parameters must be `name: Type`, and the last must be `cpu: impl Cpu`",
        );
    };

    (
        @expand
        [[$($outer:tt)*] [$($entry:tt)*] $vis:vis [$($unsafe:tt)*] $name:ident ($($ret:ty)?) $body:block];
        [$($arg:ident: $ty:ty,)*];
        $cpu:ident: impl $cpu_trait:ident
    ) => {
        $($outer)*
        #[inline(always)]
        $vis $($unsafe)* fn $name($($arg: $ty,)* $cpu: impl $cpu_trait) $(-> $ret)? {
            #[inline(always)]
            $($unsafe)* fn __impl($($arg: $ty,)* $cpu: impl $cpu_trait) $(-> $ret)? $body

            $($entry)*
            $($unsafe)* fn __generic($($arg: $ty,)* $cpu: crate::pipeline::cpu::Generic) $(-> $ret)? {
                __impl($($arg,)* $cpu)
            }

            cfg_select! {
                target_arch = "x86_64" => {
                    $($entry)*
                    #[target_feature(enable = "avx2,bmi2,popcnt")]
                    $($unsafe)* fn __avx2($($arg: $ty,)* $cpu: crate::pipeline::cpu::Avx2) $(-> $ret)? {
                        __impl($($arg,)* $cpu)
                    }

                    match crate::pipeline::cpu::Cpu::to_dyn($cpu) {
                        // SAFETY: `Avx2` type guarantees the CPU supports the features `__avx2` uses
                        crate::pipeline::cpu::DynCpu::Avx2($cpu) => unsafe { __avx2($($arg,)* $cpu) },
                        crate::pipeline::cpu::DynCpu::Generic($cpu) => __generic($($arg,)* $cpu),
                    }
                }
                target_arch = "aarch64" => {
                    $($entry)*
                    #[target_feature(enable = "neon")]
                    $($unsafe)* fn __neon($($arg: $ty,)* $cpu: crate::pipeline::cpu::Neon) $(-> $ret)? {
                        __impl($($arg,)* $cpu)
                    }

                    match crate::pipeline::cpu::Cpu::to_dyn($cpu) {
                        // SAFETY: `Neon` type guarantees the CPU supports the features `__neon` uses
                        crate::pipeline::cpu::DynCpu::Neon($cpu) => unsafe { __neon($($arg,)* $cpu) },
                        crate::pipeline::cpu::DynCpu::Generic($cpu) => __generic($($arg,)* $cpu),
                    }
                }
                _ => {
                    match crate::pipeline::cpu::Cpu::to_dyn($cpu) {
                        crate::pipeline::cpu::DynCpu::Generic($cpu) => __generic($($arg,)* $cpu),
                    }
                }
            }
        }
    };
}

#[expect(unused_imports)]
pub(super) use cpu_versions;

/// Re-exported so call sites can import it from here, along with [`cpu_versions!`].
#[expect(unused_imports)]
pub(super) use macro_rules_attr::apply;

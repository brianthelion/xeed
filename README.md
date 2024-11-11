`xeed.py` is an extremely compact project bootstrap -- "seed" -- that only
depends on Python stdlib.

`xeed.cfg` is the templated configuration file that allows `xeed.py` to be both
compact and extensible.

# Quick start
```
git clone https://github.com/brianthelion/xeed my-project && cd my-project
nano xeed.cfg # Add a "tool.hello" section with "cmdstr: echo 'Hello world'"
./xeed.py hello
```

# Intro
The main goal with this little utility was to make a highly portable project
bootstrap that only depends on two files -- one to configure it, and one to run
it. The result is a project entrypoint that only depends on the Python stdlib
but is sufficiently rich to replace large collections of complex `bash` scripts
or otherwise rehouse them. `xeed.py` itself is just a clever runner, and it
hands off its work to the tools configured in `xeed.cfg` with help from some
simple custom templating.

A secondary goal here was to support `docker` as the main tool configured by
default. This is entirely optional; you can configure the utility to hand off to
whatever secondary executables you want to.

# Configuration
`xeed.cfg` uses standard `configparser` syntax with a few tweaks.

Untweaked `configparser` supports an "extended interpolation" mode that allows
values in config file to reference itself by using placeholders, making it
easier to define variables dynamically. When extended interpolation is enabled,
you can include placeholders in the format `${section:option}` that refer to
values in other sections or the same section. This adds flexibility to
configuration files, as you can build dependencies between settings without
hard-coding values multiple times. If a placeholder is found, `configparser`
will replace it with the referenced value during parsing, allowing for
streamlined and modular configuration management.

Our modified version aims to go a little further, but we haven't quite gotten
there yet. As of now, all we've done is change the placeholder syntax to
`{section.option}`.

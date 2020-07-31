#! /usr/bin/env python
import click


@click.group()
@click.option("--debug/--no-debug", default=False)
def cli(debug):
    click.echo("Debug mode is %s" % ("on" if debug else "off"))


@click.option("--debug/--no-debug", default=False)
@cli.command()  # @cli, not @click!
def sync(debug):
    click.echo("Syncing")
    click.echo(debug)


if __name__ == "__main__":
    cli()

import sys
import os
from pathlib import Path

# Set working directory to repo root
target_dir = Path(__file__).resolve().parent.parent
os.chdir(target_dir)

# Add src to path
sys.path.insert(0, str(target_dir / "src"))

from strap.agent import (
    create_dissolve_agent,
    _extract_text,
    _show_startup_animation,
    _get_cli_version,
    _get_or_create_thread_id,
)
from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv()

    from rich.console import Console
    from rich.markdown import Markdown
    from rich.spinner import Spinner as RichSpinner
    from rich.live import Live
    from rich.text import Text

    console = Console(stderr=True)
    out = Console()  # stdout for answers

    thread_id = _get_or_create_thread_id()
    version_text = _get_cli_version()

    # ── Show the DISSOLVE startup banner ──
    _show_startup_animation(
        console,
        version_text=version_text,
        thread_id=thread_id,
        use_checkpointer=True,
    )

    # ── Load agent with spinner ──
    with Live(
        RichSpinner("dots", text=Text("Loading agent…", style="dim")),
        console=console,
        transient=True,
    ):
        agent = create_dissolve_agent(enable_persistence=True)

    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 150}

    console.print("[bold cyan]Ready![/] Type your query below. Type [bold]quit[/] to exit.\n")

    # ── Interactive REPL ──
    import time

    while True:
        try:
            user_input = out.input("[bold]> [/]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/]")
            break

        if not user_input or user_input.lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye![/]")
            break

        t0 = time.time()
        with Live(
            RichSpinner("dots", text=Text("Thinking…", style="dim")),
            console=console,
            transient=True,
        ):
            try:
                result = agent.invoke(
                    {"messages": [{"role": "user", "content": user_input}]},
                    config=config,
                )
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted.[/]\n")
                continue
            except Exception as e:
                console.print(f"\n[red]Error:[/] {e}\n")
                continue

        elapsed = time.time() - t0

        answer = None
        for msg in reversed(result["messages"]):
            if hasattr(msg, "content") and msg.type == "ai" and msg.content:
                answer = _extract_text(msg.content)
                break

        if answer:
            out.print()
            out.print(Markdown(answer))
            console.print(f"\n[dim]({elapsed:.1f}s)[/]\n")
        else:
            console.print("\n[dim]No response.[/]\n")

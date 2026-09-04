import multiprocessing as mp
import time

from cli import parse_args
from gui_worker import gui_process
from model_worker import model_process
from shared import load_prompt, put_drop


def main():
    cli = parse_args()
    cfg = vars(cli)
    cfg["prompt"] = load_prompt(cli)
    cfg["cols"] = max(1, cfg["cols"])
    cfg["rows"] = max(1, cfg["rows"])
    cfg["groups"] = max(1, cfg["groups"])
    mp.set_start_method("spawn", force=True)

    out_q = mp.Queue(maxsize=max(1, cli.queue_size))
    ctrl_q = mp.Queue(maxsize=16)
    shutdown_event = mp.Event()

    producer = mp.Process(target=model_process, args=(out_q, ctrl_q, shutdown_event, cfg), daemon=False)
    gui = mp.Process(target=gui_process, args=(out_q, ctrl_q, shutdown_event, cfg), daemon=False)
    producer.start()
    gui.start()

    try:
        producer_reported = False
        while gui.is_alive():
            if not producer.is_alive() and not producer_reported:
                producer.join(timeout=0.1)
                producer_reported = True
                if producer.exitcode == 0:
                    put_drop(out_q, ("done", "Generation process finished; window remains open."))
                    print("demo4: producer exited normally; keeping GUI open.", flush=True)
                else:
                    put_drop(out_q, ("error", f"producer exited with code {producer.exitcode}"))
                    print(f"demo4: producer exited with code {producer.exitcode}; keeping GUI open for diagnostics.", flush=True)
            time.sleep(0.2)
    except KeyboardInterrupt:
        shutdown_event.set()
    finally:
        shutdown_event.set()
        for name, proc in (("producer", producer), ("gui", gui)):
            proc.join(timeout=5.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2.0)
            if proc.is_alive():
                proc.kill()
                proc.join()
            print(f"demo4: {name} exitcode={proc.exitcode}", flush=True)


if __name__ == "__main__":
    main()

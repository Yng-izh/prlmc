#!/usr/bin/env python3
"""Install the PRLM-C adapter into the copied benchmark tree.

Run from the root of the PRLM-C copy:

    python apply_patch.py [--root /share/latent/PRLM-C] [--check]

Two host files are edited, both idempotently:

  agent.py         adds a `prlmc` branch to _initialize_agent_by_type and to
                   send_message. Both go *before* the `rag` branch because
                   _is_agent_type does substring matching.
  utils/templates.py  adds 'prlmc': 'rag_agent' to AGENT_TYPE_MAPPING so
                   get_template resolves a query template for the agent.

A .bak copy is written next to each file the first time it is modified. Nothing
else in the tree is touched; methods/prlmc/ is new code and is not patched in.
"""

import argparse
import os
import shutil
import sys

INIT_MARKER = "methods.prlmc.agent"
SEND_MARKER = "handle_prlmc_agent"
TEMPLATE_KEY = "'prlmc': 'rag_agent',"


def _read(path):
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _write(path, text, backup=True):
    if backup and not os.path.exists(path + ".bak"):
        shutil.copy2(path, path + ".bak")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def patch_agent(root, check=False):
    path = os.path.join(root, "agent.py")
    text = _read(path)
    original = text

    if INIT_MARKER not in text:
        anchor = """        elif self._is_agent_type("memoryllm"):
            from methods.memoryllm import initialize_memoryllm_agent
            initialize_memoryllm_agent(self, agent_config)
"""
        addition = """        elif self._is_agent_type("prlmc"):
            from methods.prlmc.agent import initialize_prlmc_agent
            initialize_prlmc_agent(self, agent_config)
"""
        if anchor not in text:
            raise SystemExit(
                "agent.py: could not find the memoryllm initialization branch; "
                "PRLM-C expects the MemoryLLM baseline to be present"
            )
        text = text.replace(anchor, anchor + addition, 1)

    if SEND_MARKER not in text:
        anchor = """        elif self._is_agent_type("memoryllm"):
            from methods.memoryllm import handle_memoryllm_agent
            return handle_memoryllm_agent(self, message, memorizing)
"""
        addition = """        elif self._is_agent_type("prlmc"):
            from methods.prlmc.agent import handle_prlmc_agent
            return handle_prlmc_agent(self, message, memorizing)
"""
        if anchor not in text:
            raise SystemExit("agent.py: could not find the memoryllm send_message branch")
        text = text.replace(anchor, anchor + addition, 1)

    if text == original:
        print("agent.py: already patched")
        return False
    if check:
        print("agent.py: WOULD patch")
        return True
    _write(path, text)
    print("agent.py: patched (backup at agent.py.bak)")
    return True


def patch_templates(root, check=False):
    path = os.path.join(root, "utils", "templates.py")
    text = _read(path)
    original = text

    if TEMPLATE_KEY not in text:
        anchor = "AGENT_TYPE_MAPPING = {\n"
        if anchor not in text:
            raise SystemExit("templates.py: AGENT_TYPE_MAPPING not found")
        text = text.replace(anchor, anchor + "    " + TEMPLATE_KEY + "\n", 1)

    if text == original:
        print("utils/templates.py: already patched")
        return False
    if check:
        print("utils/templates.py: WOULD patch")
        return True
    _write(path, text)
    print("utils/templates.py: patched (backup at utils/templates.py.bak)")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="root of the PRLM-C tree")
    parser.add_argument("--check", action="store_true", help="report without writing")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isfile(os.path.join(root, "agent.py")):
        raise SystemExit(f"{root} does not look like the benchmark tree (no agent.py)")

    changed = patch_agent(root, args.check)
    changed |= patch_templates(root, args.check)
    print("changes applied" if changed else "nothing to do")
    return 0


if __name__ == "__main__":
    sys.exit(main())

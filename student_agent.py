import re
from collections import deque


class AssemblyAgent:

    def __init__(self):
        pass

    # ============================================================
    # PARSE
    # ============================================================

    def parse_blocks(self, scenario_context):

        statements = scenario_context.split("[STATEMENT]")
        last_statement = statements[-1]

        if "[PLAN]" in last_statement:
            last_statement = last_statement.split("[PLAN]")[0]

        if "My goal is" not in last_statement:
            return set(), [], []

        init_part = last_statement.split("My goal is")[0]
        goal_part = last_statement.split("My goal is")[1]

        init_match = re.search(r"that, (.*)", init_part, re.DOTALL)
        goal_match = re.search(r"that (.*)", goal_part, re.DOTALL)

        if not init_match or not goal_match:
            return set(), [], []

        initial_text = init_match.group(1).strip().rstrip(".")
        goal_text = goal_match.group(1).strip().rstrip(".")

        initial_items = [x.strip() for x in initial_text.split(",")]
        goal_items = [x.strip() for x in goal_text.split(",")]

        blocks = set()
        for item in initial_items + goal_items:
            matches = re.findall(r"the (\w+) block", item)
            for m in matches:
                blocks.add(m)

        return blocks, initial_items, goal_items

    # ============================================================
    # BUILD STATE
    # ============================================================

    def build_state(self, blocks, initial_items):

        on = {}
        holding = None

        for fact in initial_items:
            if "is on top of" in fact:
                matches = re.findall(r"the (\w+) block", fact)
                if len(matches) >= 2:
                    on[matches[0]] = matches[1]

        supported = set(on.keys())
        on_table = {b for b in blocks if b not in supported}

        clear = {b: True for b in blocks}
        for a, b in on.items():
            clear[b] = False

        return {
            "on": on,
            "holding": holding,
            "clear": clear,
            "on_table": on_table
        }

    # ============================================================
    # GOAL
    # ============================================================

    def is_goal(self, state, goal_items):

        for fact in goal_items:
            if "is on top of" in fact:
                matches = re.findall(r"the (\w+) block", fact)
                if len(matches) >= 2:
                    if state["on"].get(matches[0]) != matches[1]:
                        return False

        return True

    # ============================================================
    # ACTIONS
    # ============================================================

    def possible_actions(self, state, blocks):

        actions = []

        if state["holding"] is None:
            for b in blocks:
                if b in state["on_table"] and state["clear"][b]:
                    actions.append(("pick", b))

            for a, b in state["on"].items():
                if state["clear"][a]:
                    actions.append(("unmount", a, b))

        else:
            actions.append(("put", state["holding"]))
            for b in blocks:
                if state["clear"][b]:
                    actions.append(("mount", state["holding"], b))

        return actions

    # ============================================================
    # APPLY
    # ============================================================

    def apply_action(self, state, action):

        new_state = {
            "on": dict(state["on"]),
            "holding": state["holding"],
            "clear": dict(state["clear"]),
            "on_table": set(state["on_table"])
        }

        if action[0] == "pick":
            b = action[1]
            new_state["holding"] = b
            new_state["on_table"].discard(b)
            new_state["clear"][b] = False

        elif action[0] == "unmount":
            a, b = action[1], action[2]
            new_state["holding"] = a
            new_state["on"].pop(a)
            new_state["clear"][b] = True
            new_state["clear"][a] = False

        elif action[0] == "put":
            b = action[1]
            new_state["holding"] = None
            new_state["on_table"].add(b)
            new_state["clear"][b] = True

        elif action[0] == "mount":
            a, b = action[1], action[2]
            new_state["holding"] = None
            new_state["on"][a] = b
            new_state["clear"][b] = False
            new_state["clear"][a] = True

        return new_state

    # ============================================================
    # FORMAT
    # ============================================================

    def format_action(self, action):

        if action[0] == "pick":
            return f"pick up the {action[1]} block"
        if action[0] == "unmount":
            return f"unmount_node the {action[1]} block from on top of the {action[2]} block"
        if action[0] == "put":
            return f"put down the {action[1]} block"
        if action[0] == "mount":
            return f"mount_node the {action[1]} block on top of the {action[2]} block"

    # ============================================================
    # BFS
    # ============================================================

    def bfs(self, initial_state, goal_items, blocks):

        queue = deque([(initial_state, [])])
        visited = set()

        while queue:
            state, path = queue.popleft()

            state_key = (
                tuple(sorted(state["on"].items())),
                state["holding"],
                tuple(sorted(state["clear"].items())),
                tuple(sorted(state["on_table"]))
            )

            if state_key in visited:
                continue

            visited.add(state_key)

            if self.is_goal(state, goal_items):
                return path

            for action in self.possible_actions(state, blocks):
                new_state = self.apply_action(state, action)
                queue.append((new_state, path + [action]))

        return []

    # ============================================================
    # SOLVE COMPATIBLE CON DEV Y SUBMIT
    # ============================================================

    def solve(self, *args):

        # Caso dev_test.py
        if len(args) == 2 and isinstance(args[0], str):
            scenario_context = args[0]
            plan = self._solve_internal(scenario_context)
            return plan

        # Caso submit.py
        if len(args) == 2:
            assembly_task_id, scenario_context = args
            plan = self._solve_internal(scenario_context)
            return {
                "complexity_level": len(plan),
                "target_action_sequence": plan
            }

        return []

    def _solve_internal(self, scenario_context):

        blocks, initial_items, goal_items = self.parse_blocks(scenario_context)

        if not blocks:
            return []

        initial_state = self.build_state(blocks, initial_items)
        plan = self.bfs(initial_state, goal_items, blocks)

        return [self.format_action(a) for a in plan]
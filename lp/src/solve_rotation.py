from ortools.linear_solver import pywraplp
from utils import write_solution

from lp.src.solve import LPsolver


class LPsolverRot(LPsolver):

    def __init__(self, data, output_dir, timeout):
        super().__init__(data, output_dir, timeout)
        if output_dir == "":
            output_dir = "./lp/out/rot"
        self.output_dir = output_dir

    def solve_instance(self, instance):
        _, self.max_width, self.circuits = instance
        self.circuits_num = len(self.circuits)

        w, h = ([i for i, _ in self.circuits], [j for _, j in self.circuits])

        lower_bound = sum([h[i] * w[i] for i in range(self.circuits_num)]) // self.max_width
        upper_bound = sum(h) - min(h)

        # creating the model
        solver = pywraplp.Solver.CreateSolver('BOP')
        solver.SetTimeLimit(self.timeout * 1000)
        solver.SetNumThreads(8)

        max_h = sum(h)

        x = []
        y = []
        widths = []
        heights = []

        H = solver.IntVar(lb=lower_bound, ub=upper_bound, name='h')
        for i in range(self.circuits_num):
            widths.append(solver.IntVar(lb=min(w), ub=max(w), name=f'widths_{i}'))
            heights.append(solver.IntVar(lb=min(h), ub=max(h), name=f'heights_{i}'))
            x.append(solver.IntVar(lb=0, ub=self.max_width - w[i], name=f'x_{i}'))
            y.append(solver.IntVar(lb=0, ub=max_h - h[i], name=f'y_{i}'))

        rot = []
        for i in range(self.circuits_num):
            rot.append(solver.IntVar(lb=0, ub=1, name=f'rot_{i}'))
            solver.Add(widths[i] == w[i] - rot[i] * w[i] + rot[i] * h[i])
            solver.Add(heights[i] == h[i] - rot[i] * h[i] + rot[i] * w[i])

        # disjunction bool variables
        d = [
            [[solver.IntVar(lb=0, ub=1, name=f'd_{i}_{j}_{k}') for k in range(4)] for j in range(self.circuits_num)]
            for i in range(self.circuits_num)]
        M = 1000

        for i in range(self.circuits_num):
            # domain constraints
            solver.Add(x[i] + widths[i] <= self.max_width)
            solver.Add(y[i] + heights[i] <= max_h)
            solver.Add(H >= y[i] + heights[i])
            # non overlapping constraints
            for j in range(i + 1, self.circuits_num):
                solver.Add(sum(d[i][j]) >= 1)
                solver.Add(x[i] + widths[i] <= x[j] + M * (1 - d[i][j][0]))
                solver.Add(x[j] + widths[j] <= x[i] + M * (1 - d[i][j][1]))
                solver.Add(y[i] + heights[i] <= y[j] + M * (1 - d[i][j][2]))
                solver.Add(y[j] + heights[j] <= y[i] + M * (1 - d[i][j][3]))

        solver.Minimize(H)
        status = solver.Solve()
        total_time = solver.WallTime() / 1000

        if status == pywraplp.Solver.OPTIMAL:
            circuit_pos = [(int(w), int(h), int(x), int(y)) for w, h, x, y in
                           zip([a.solution_value() for a in widths], [a.solution_value() for a in heights],
                               [a.solution_value() for a in x], [a.solution_value() for a in y])]
            write_solution(self.output_dir, self.ins_num, ((self.max_width, int(H.solution_value())), circuit_pos),
                           total_time)
            return self.ins_num, ((self.max_width, int(H.solution_value())), circuit_pos), total_time
        else:
            write_solution(self.output_dir, self.ins_num, None, 0)
            return self.ins_num, None, 0

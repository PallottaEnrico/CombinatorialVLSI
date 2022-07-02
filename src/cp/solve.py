import datetime

from minizinc import Model, Solver, Status, Instance
import numpy as np

from utils.utils import write_solution


class CPsolver:

    def __init__(self, data, rotation, output_dir, timeout):
        self.data = data
        self.rotation = rotation
        if output_dir == "":
            output_dir = "../data/output_cp/"
        self.output_dir = output_dir
        self.timeout = timeout
        if rotation:
            self.solver_path = ".\\cp\\model_with_rotations.mzn"
        else:
            self.solver_path = ".\\cp\\model.mzn"

    def solve(self):
        model = Model(self.solver_path)
        gecode = Solver.lookup("gecode")
        for d in self.data:
            ins_num, plate_width, circuits = d
            instance = Instance(gecode, model)
            instance["N"] = len(circuits)
            instance["W"] = plate_width

            instance["widths"] = [x for (x, _) in circuits]
            instance["heights"] = [y for (_, y) in circuits]

            result = instance.solve(timeout=datetime.timedelta(seconds=self.timeout))

            if result.status is Status.OPTIMAL_SOLUTION:
                if self.rotation:
                    circuits_pos = [(w, h, x, y) if not r else (h, w, x, y) for (w, h), x, y, r in
                                    zip(circuits, result["x"], result["y"], result["r"])]
                else:
                    circuits_pos = [(w, h, x, y) for (w, h), x, y in
                                    zip(circuits, result["coords_x"], result["coords_y"])]
                plate_height = result.objective

                write_solution(ins_num, ((plate_width, plate_height), circuits_pos),
                               result.statistics['time'].total_seconds())
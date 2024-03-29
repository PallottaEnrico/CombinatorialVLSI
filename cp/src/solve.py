import datetime

from minizinc import Model, Solver, Status, Instance

from utils import write_solution


class CPsolver:

    def __init__(self, data, rotation, output_dir, timeout):
        self.data = data
        self.rotation = rotation
        if output_dir == "":
            output_dir = "./cp/out/rot" if rotation else "./cp/out/no_rot"
        self.output_dir = output_dir
        self.timeout = timeout
        if rotation:
            self.solver_path = "./cp/src/models/model_with_rotations.mzn"
        else:
            self.solver_path = "./cp/src/models/model.mzn"

    def solve(self):
        model = Model(self.solver_path)
        solver = Solver.lookup("chuffed")

        solutions = []

        for d in self.data:
            try:
                ins_num, plate_width, circuits = d
                instance = Instance(solver, model)
                instance["N"] = len(circuits)
                instance["W"] = plate_width
                instance["w"] = [x for (x, _) in circuits]
                instance["h"] = [y for (_, y) in circuits]

                result = instance.solve(timeout=datetime.timedelta(seconds=self.timeout), processes=10, random_seed=42,
                                        free_search=True)

                if result.status is Status.OPTIMAL_SOLUTION:
                    if self.rotation:
                        circuits_pos = [(w, h, x, y) if not r else (h, w, x, y) for (w, h), x, y, r in
                                        zip(circuits, result["x"], result["y"], result["rotation"])]
                    else:
                        circuits_pos = [(w, h, x, y) for (w, h), x, y in
                                        zip(circuits, result["x"], result["y"])]
                    plate_height = result.objective

                    write_solution(self.output_dir, ins_num, ((plate_width, plate_height), circuits_pos),
                                   result.statistics['time'].total_seconds())

                    solutions.append((ins_num, ((plate_width, plate_height), circuits_pos),
                                      result.statistics['time'].total_seconds()))
            except:
                # If no solution is found in timeout seconds,
                # do nothing and pass to the next instance.
                pass

        return solutions

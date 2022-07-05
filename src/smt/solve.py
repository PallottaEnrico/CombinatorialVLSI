import time

from utils.utils import write_solution
from z3 import And, Or, sat, Solver, Sum, If, IntVector, Implies


class SMTsolver:

    def __init__(self, data, output_dir, rotation=False, timeout=300):
        self.data = data
        self.rotation = rotation
        if output_dir == "":
            output_dir = "../data/output_smt/"
        self.output_dir = output_dir
        self.timeout = timeout

        self.circuits_num = None
        self.circuits = None
        self.max_width = None
        self.y_positions = None
        self.x_positions = None
        self.sol = None
        self.h = None
        self.w = None

    def solve(self):
        solutions = []
        for d in self.data:
            ins_num = d[0]
            print(f"Instance num: {ins_num}")
            solutions.append(self.solve_instance(d, ins_num))
        return solutions

    def solve_instance(self, instance, ins_num):
        _, self.max_width, self.circuits = instance
        self.circuits_num = len(self.circuits)

        self.w = IntVector('widths', self.circuits_num)
        self.h = IntVector('heights', self.circuits_num)

        for i in range(self.circuits_num):
            self.w[i], _ = self.circuits[i]
            _, self.h[i] = self.circuits[i]

        #        self.w, self.h = ([i for i, _ in self.circuits], [j for _, j in self.circuits])

        lower_bound = sum([self.h[i] * self.w[i] for i in range(self.circuits_num)]) // self.max_width
        upper_bound = sum(self.h) - min(self.h)

        for plate_height in range(lower_bound, upper_bound + 1):
            self.sol = Solver()
            self.sol.set(timeout=self.timeout * 1000)
            #            plate, rotations = self.set_constraints(plate_height)

            self.set_constraints(plate_height)

            solve_time = time.time()
            if self.sol.check() == sat:
                spent_time = time.time() - solve_time
                circuits_pos = self.evaluate()
                write_solution(self.output_dir, ins_num, ((self.max_width, plate_height), circuits_pos),
                               spent_time)
                return ins_num, ((self.max_width, plate_height), circuits_pos), spent_time
            else:
                try_timeout = round((self.timeout - (time.time() - solve_time)))
                if try_timeout < 0:
                    return None, 0
        return None, 0

    def set_constraints(self, plate_height):

        self.x_positions = IntVector('x_pos', self.circuits_num)
        self.y_positions = IntVector('y_pos', self.circuits_num)

        # actual widths and heights if rotation allowed
        w, h = self.w, self.h

        # Handling rotation
        if self.rotation:
            for i in range(self.circuits_num):
                # if rotation allowed and a circuit is square  (width=height) then force it to be not rotated
                self.sol.add(If(w[i] == h[i], And(w[i] == self.w[i], h[i] == self.h[i]),
                                Or(And(w[i] == self.w[i], h[i] == self.h[i]),
                                   And(w[i] == self.h[i], h[i] == self.w[i]))))

        # CONSTRAINTS

        # NOT EXCEED, DOMAINS
        self.sol.add([And(0 <= self.x_positions[i], self.x_positions[i] <= self.max_width - self.w[i])
                      for i in range(self.circuits_num)])

        self.sol.add([And(0 <= self.y_positions[i], self.y_positions[i] <= plate_height - self.h[i])
                      for i in range(self.circuits_num)])

        # DO NOT OVERLAP
        for i in range(1, self.circuits_num):
            for j in range(0, i):
                self.sol.add(Or(Sum(self.y_positions[i], self.h[i]) <= self.y_positions[j],
                                Sum(self.y_positions[j], self.h[j]) <= self.y_positions[i],
                                Sum(self.x_positions[i], self.w[i]) <= self.x_positions[j],
                                Sum(self.x_positions[j], self.w[j]) <= self.x_positions[i]))

                # Two rectangles with same dimensions
                self.sol.add(Implies(And(self.w[i] == self.w[j], self.h[i] == self.h[j]),
                                     Or(self.x_positions[j] > self.x_positions[i],
                                        And(self.x_positions[j] == self.x_positions[i],
                                            self.y_positions[j] >= self.y_positions[i]))))

                # If two rectangles cannot be packed side to side along the x axis
                self.sol.add(Implies(Sum(self.w[i], self.w[j]) > self.max_width,
                                     Or(Sum(self.y_positions[i], self.h[i]) <= self.y_positions[j],
                                        Sum(self.y_positions[j], self.h[j]) <= self.y_positions[i])))

                # If two rectangles cannot be packed one over the other along the y axis
                self.sol.add(Implies(Sum(self.h[i], self.h[j]) > plate_height,
                                     Or(Sum(self.x_positions[i], self.w[i]) <= self.x_positions[j],
                                        Sum(self.x_positions[j],
                                            self.w[j]) <= self.x_positions[i])))

        # SUM OVER ROWS (CUMULATIVE)
        for u in range(plate_height):
            self.sol.add(
                self.max_width >= Sum([If(And(self.y_positions[i] <= u, u < Sum(self.y_positions[i], self.h[i])),
                                          self.w[i], 0) for i in range(self.circuits_num)]))

        # SUM OVER COLUMNS (CUMULATIVE)
        for u in range(self.max_width):
            self.sol.add(plate_height >= Sum([If(And(self.x_positions[i] <= u, u < Sum(self.x_positions[i], self.w[i])),
                                                 self.h[i], 0) for i in range(self.circuits_num)]))

    def evaluate(self):
        return [(self.w[i], self.h[i], int(self.sol.model().evaluate(self.x_positions[i]).as_string()),
                 int(self.sol.model().evaluate(self.y_positions[i]).as_string())) for i in range(self.circuits_num)]
from .motor import MotorNominal, MotorPlant
from .reactor import ReactorNominal, ReactorPlant
from .robot import RobotNominal, RobotPlant


def make(spec, rng, mismatch):
    """Return (generator plant, detector-side nominal model) for a run."""
    s = spec["scenario"]
    if s == "motor":
        return MotorPlant(spec, rng, mismatch), MotorNominal()
    if s == "reactor":
        return ReactorPlant(spec, rng, mismatch), ReactorNominal()
    if s == "robot":
        plant = RobotPlant(spec, rng, mismatch)
        return plant, RobotNominal(plant.joint_of)       # sensor-to-joint wiring is known
    raise ValueError(s)

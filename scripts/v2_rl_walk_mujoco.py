import time
import pickle

import numpy as np
from mini_bdx_runtime.rustypot_position_hwi import HWI
from mini_bdx_runtime.onnx_infer import OnnxInfer

from mini_bdx_runtime.raw_imu import Imu
from mini_bdx_runtime.poly_reference_motion import PolyReferenceMotion
from mini_bdx_runtime.xbox_controller import XBoxController
from mini_bdx_runtime.feet_contacts import FeetContacts
from mini_bdx_runtime.eyes import Eyes
from mini_bdx_runtime.sounds import Sounds
from mini_bdx_runtime.antennas import Antennas
from mini_bdx_runtime.projector import Projector
from mini_bdx_runtime.rl_utils import make_action_dict, LowPassActionFilter
from mini_bdx_runtime.duck_config import DuckConfig

import os

HOME_DIR = os.path.expanduser("~")


class RLWalk:
    def __init__(
        self,
        onnx_model_path: str,
        duck_config_path: str = f"{HOME_DIR}/duck_config.json",
        serial_port: str = "/dev/ttyACM0",
        control_freq: float = 50,
        pid=[30, 0, 0],
        action_scale=0.25,
        commands=False,
        pitch_bias=0,
        save_obs=False,
        replay_obs=None,
        cutoff_frequency=None,
        head_animation=False,
        animation_clip=None,
        animation_derating=0.5,
    ):

        self.duck_config = DuckConfig(config_json_path=duck_config_path)

        self.commands = commands
        self.pitch_bias = pitch_bias

        self.onnx_model_path = onnx_model_path
        self.policy = OnnxInfer(self.onnx_model_path, awd=True)

        self.num_dofs = 14
        self.max_motor_velocity = 5.24  # rad/s

        # Control
        self.control_freq = control_freq
        self.pid = pid

        self.save_obs = save_obs
        if self.save_obs:
            self.saved_obs = []

        self.replay_obs = replay_obs
        if self.replay_obs is not None:
            self.replay_obs = pickle.load(open(self.replay_obs, "rb"))

        self.action_filter = None
        if cutoff_frequency is not None:
            self.action_filter = LowPassActionFilter(
                self.control_freq, cutoff_frequency
            )

        self.hwi = HWI(self.duck_config, serial_port)

        self.start()

        self.imu = Imu(
            sampling_freq=int(self.control_freq),
            user_pitch_bias=self.pitch_bias,
            upside_down=self.duck_config.imu_upside_down,
        )

        self.feet_contacts = FeetContacts()

        # Scales
        self.action_scale = action_scale

        self.last_action = np.zeros(self.num_dofs)
        self.last_last_action = np.zeros(self.num_dofs)
        self.last_last_last_action = np.zeros(self.num_dofs)

        self.init_pos = list(self.hwi.init_pos.values())

        self.motor_targets = np.array(self.init_pos.copy())
        self.prev_motor_targets = np.array(self.init_pos.copy())

        self.last_commands = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.paused = self.duck_config.start_paused

        self.command_freq = 20  # hz
        if self.commands:
            self.xbox_controller = XBoxController(self.command_freq)

        # Reference motion, but we only really need the length of one phase
        # TODO
        self.PRM = PolyReferenceMotion("./polynomial_coefficients.pkl")
        self.imitation_i = 0
        self.imitation_phase = np.array([0, 0])
        self.phase_frequency_factor = 1.0
        self.phase_frequency_factor_offset = (
            self.duck_config.phase_frequency_factor_offset
        )

        # Optional expression features
        if self.duck_config.eyes:
            self.eyes = Eyes()
        if self.duck_config.projector:
            self.projector = Projector()
        if self.duck_config.speaker:
            self.sounds = Sounds(
                volume=1.0, sound_directory="../mini_bdx_runtime/assets/"
            )
        if self.duck_config.antennas:
            self.antennas = Antennas()

        # --- Phase 4: optional additive head animation overlay ---------------
        # While the RL policy owns the legs (STAND/WALK), the head is driven by
        # the deployed additive path (see the head_motor_targets lines in run()).
        # When enabled, an open_duck_anim.Engine is evaluated EXACTLY ONCE per
        # control tick from the SAME loop clock (no second timer thread — a
        # separate animation clock would drift against the policy clock), and its
        # envelope-clamped head offsets are routed into last_commands[3:7] so
        # they flow through that existing additive path. Legs are never touched.
        self.head_animation = head_animation
        self.anim_engine = None
        self.anim_show = None
        self._anim_t0 = None
        if self.head_animation:
            from mini_bdx_runtime.anim.controller import make_head_engine
            from open_duck_anim import Triggers as _AnimTriggers, load_clip as _load_clip

            self._AnimTriggers = _AnimTriggers
            bg = None
            if animation_clip is not None:
                bg = _load_clip(animation_clip)
            # Head output ALWAYS passes the measured safety envelope; derated
            # (x0.5 by default) for early hardware trials — relax as data accrues.
            self.anim_engine = make_head_engine(
                envelope_derating=animation_derating, background_clip=bg
            )
            self._anim_t0 = time.monotonic()
            print(
                f"[anim] head animation ENABLED (derating={animation_derating}, "
                f"clip={'<background>' if bg is None else animation_clip})"
            )

    def get_obs(self):

        imu_data = self.imu.get_data()

        dof_pos = self.hwi.get_present_positions(
            ignore=[
                "left_antenna",
                "right_antenna",
            ]
        )  # rad

        dof_vel = self.hwi.get_present_velocities(
            ignore=[
                "left_antenna",
                "right_antenna",
            ]
        )  # rad/s

        if dof_pos is None or dof_vel is None:
            return None

        if len(dof_pos) != self.num_dofs:
            print(f"ERROR len(dof_pos) != {self.num_dofs}")
            return None

        if len(dof_vel) != self.num_dofs:
            print(f"ERROR len(dof_vel) != {self.num_dofs}")
            return None

        cmds = self.last_commands

        feet_contacts = self.feet_contacts.get()

        obs = np.concatenate(
            [
                imu_data["gyro"],
                imu_data["accelero"],
                cmds,
                dof_pos - self.init_pos,
                dof_vel * 0.05,
                self.last_action,
                self.last_last_action,
                self.last_last_last_action,
                self.motor_targets,
                feet_contacts,
                self.imitation_phase,
            ]
        )

        return obs

    def _drive_anim_show(self, show):
        """Route the animation engine's show tick to the expression hardware.

        Antennas are driven ONLY from ``show`` (never the joint array) via the
        existing ``antennas.py`` PWM path (D13 left +1 / D12 right -1). Discrete
        sound/projector events fire exactly once (the engine edge-triggers them).
        Eyes keep their existing autonomous blink thread in the RL loop.
        """
        if show is None:
            return
        if self.duck_config.antennas:
            # show.antenna_l / antenna_r are already per-side normalised [-1, 1]
            # with the clip's left=+1 / right=-1 calibration baked in.
            self.antennas.set_position_left(float(show.antenna_l))
            self.antennas.set_position_right(float(show.antenna_r))
        for ev in show.events:
            if ev.type == "sound" and self.duck_config.speaker:
                self.sounds.play(ev.value)
            elif ev.type == "projector" and self.duck_config.projector:
                want_on = str(ev.value).lower() in ("on", "1", "true")
                if want_on != self.projector.on:
                    self.projector.switch()

    def start(self):
        kps = [self.pid[0]] * 14
        kds = [self.pid[2]] * 14

        # lower head kps
        kps[5:9] = [8, 8, 8, 8]

        self.hwi.set_kps(kps)
        self.hwi.set_kds(kds)
        self.hwi.turn_on()

        time.sleep(2)

    def get_phase_frequency_factor(self, x_velocity):

        max_phase_frequency = 1.2
        min_phase_frequency = 1.0

        # Perform linear interpolation
        freq = min_phase_frequency + (abs(x_velocity) / 0.15) * (
            max_phase_frequency - min_phase_frequency
        )

        return freq

    def run(self):
        i = 0
        try:
            print("Starting")
            start_t = time.time()
            while True:
                left_trigger = 0
                right_trigger = 0
                t = time.time()

                if self.commands:
                    self.last_commands, self.buttons, left_trigger, right_trigger = (
                        self.xbox_controller.get_last_command()
                    )
                    if self.buttons.dpad_up.triggered:
                        self.phase_frequency_factor_offset += 0.05
                        print(
                            f"Phase frequency factor offset {round(self.phase_frequency_factor_offset, 3)}"
                        )

                    if self.buttons.dpad_down.triggered:
                        self.phase_frequency_factor_offset -= 0.05
                        print(
                            f"Phase frequency factor offset {round(self.phase_frequency_factor_offset, 3)}"
                        )

                    if self.buttons.LB.is_pressed:
                        self.phase_frequency_factor = 1.3
                    else:
                        self.phase_frequency_factor = 1.0

                    if self.buttons.X.triggered:
                        if self.duck_config.projector:
                            self.projector.switch()

                    if self.buttons.B.triggered:
                        if self.duck_config.speaker:
                            self.sounds.play_random_sound()

                    if self.duck_config.antennas:
                        self.antennas.set_position_left(right_trigger)
                        self.antennas.set_position_right(left_trigger)

                    if self.buttons.A.triggered:
                        self.paused = not self.paused
                        if self.paused:
                            print("PAUSE")
                        else:
                            print("UNPAUSE")

                if self.paused:
                    time.sleep(0.1)
                    continue

                # --- Phase 4: additive head animation (evaluate engine ONCE) ---
                # Route envelope-clamped head offsets into last_commands[3:7] so
                # they (a) enter the obs the policy sees and (b) flow through the
                # deployed additive head path below. Legs are NEVER touched here.
                if self.anim_engine is not None:
                    t_anim = time.monotonic() - self._anim_t0
                    eng = self.anim_engine.evaluate(
                        t_anim, "walk", self._AnimTriggers()
                    )
                    ho = eng.head_command_offsets  # (4,) already envelope-clamped
                    for k in range(4):
                        self.last_commands[3 + k] = (
                            float(self.last_commands[3 + k]) + float(ho[k])
                        )
                    self.anim_show = eng.show
                    self._drive_anim_show(eng.show)

                obs = self.get_obs()
                if obs is None:
                    continue

                self.imitation_i += 1 * (
                    self.phase_frequency_factor + self.phase_frequency_factor_offset
                )
                self.imitation_i = self.imitation_i % self.PRM.nb_steps_in_period
                self.imitation_phase = np.array(
                    [
                        np.cos(
                            self.imitation_i / self.PRM.nb_steps_in_period * 2 * np.pi
                        ),
                        np.sin(
                            self.imitation_i / self.PRM.nb_steps_in_period * 2 * np.pi
                        ),
                    ]
                )

                if self.save_obs:
                    self.saved_obs.append(obs)

                if self.replay_obs is not None:
                    if i < len(self.replay_obs):
                        obs = self.replay_obs[i]
                    else:
                        print("BREAKING ")
                        break

                action = self.policy.infer(obs)

                self.last_last_last_action = self.last_last_action.copy()
                self.last_last_action = self.last_action.copy()
                self.last_action = action.copy()

                # action = np.zeros(10)

                self.motor_targets = self.init_pos + action * self.action_scale

                if self.action_filter is not None:
                    self.action_filter.push(self.motor_targets)
                    filtered_motor_targets = self.action_filter.get_filtered_action()
                    if (
                        time.time() - start_t > 1
                    ):  # give time to the filter to stabilize
                        self.motor_targets = filtered_motor_targets

                # Deployed additive head path (permanent architecture): the
                # walking policy does NOT actuate the head; the head is commanded
                # additively from last_commands[3:7] (operator aim + any Phase 4
                # animation overlay). Keep these two lines.
                head_motor_targets = self.last_commands[3:] + self.motor_targets[5:9]
                self.motor_targets[5:9] = head_motor_targets

                # Phase 4 SAFETY: re-enable the max joint velocity clip, applied
                # to the FINAL 14-DOF bus targets (AFTER the additive head), not
                # to the animation commands — this is where the plan (§6.5) says
                # it belongs so no single tick can slew any joint faster than
                # max_motor_velocity regardless of policy or animation output.
                dt = 1.0 / self.control_freq
                self.motor_targets = np.clip(
                    self.motor_targets,
                    self.prev_motor_targets - self.max_motor_velocity * dt,
                    self.prev_motor_targets + self.max_motor_velocity * dt,
                )
                self.prev_motor_targets = self.motor_targets.copy()

                action_dict = make_action_dict(
                    self.motor_targets, list(self.hwi.joints.keys())
                )

                self.hwi.set_position_all(action_dict)

                i += 1

                took = time.time() - t
                # print("Full loop took", took, "fps : ", np.around(1 / took, 2))
                if (1 / self.control_freq - took) < 0:
                    print(
                        "Policy control budget exceeded by",
                        np.around(took - 1 / self.control_freq, 3),
                    )
                time.sleep(max(0, 1 / self.control_freq - took))

        except KeyboardInterrupt:
            if self.duck_config.antennas:
                self.antennas.stop()
            if self.duck_config.eyes:
                self.eyes.stop()
            if self.duck_config.projector:
                self.projector.stop()
            self.feet_contacts.stop()

        if self.save_obs:
            pickle.dump(self.saved_obs, open("robot_saved_obs.pkl", "wb"))
        print("TURNING OFF")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_model_path", type=str, required=True)
    parser.add_argument(
        "--duck_config_path",
        type=str,
        required=False,
        default=f"{HOME_DIR}/duck_config.json",
    )
    parser.add_argument("-a", "--action_scale", type=float, default=0.25)
    parser.add_argument("-p", type=int, default=30)
    parser.add_argument("-i", type=int, default=0)
    parser.add_argument("-d", type=int, default=0)
    parser.add_argument("-c", "--control_freq", type=int, default=50)
    parser.add_argument("--pitch_bias", type=float, default=0, help="deg")
    parser.add_argument(
        "--commands",
        action="store_true",
        default=True,
        help="external commands, keyboard or gamepad. Launch control_server.py on host computer",
    )
    parser.add_argument(
        "--save_obs",
        type=str,
        required=False,
        default=False,
        help="save the run's observations",
    )
    parser.add_argument(
        "--replay_obs",
        type=str,
        required=False,
        default=None,
        help="replay the observations from a previous run (can be from the robot or from mujoco)",
    )
    parser.add_argument("--cutoff_frequency", type=float, default=None)
    parser.add_argument(
        "--head_animation",
        action="store_true",
        default=False,
        help="Phase 4: overlay open_duck_anim head animation onto the additive "
        "head path while walking/standing (envelope-clamped, legs untouched).",
    )
    parser.add_argument(
        "--animation_clip",
        type=str,
        default=None,
        help="Path to a .duckanim clip used as the always-on background head "
        "loop when --head_animation is set (default: engine idle background).",
    )
    parser.add_argument(
        "--animation_derating",
        type=float,
        default=0.5,
        help="Head safety-envelope derating for --head_animation (0.5 = plan "
        "§6.5 default for early hardware trials; 1.0 = full measured envelope).",
    )

    args = parser.parse_args()
    pid = [args.p, args.i, args.d]

    print("Done parsing args")
    rl_walk = RLWalk(
        args.onnx_model_path,
        duck_config_path=args.duck_config_path,
        action_scale=args.action_scale,
        pid=pid,
        control_freq=args.control_freq,
        commands=args.commands,
        pitch_bias=args.pitch_bias,
        save_obs=args.save_obs,
        replay_obs=args.replay_obs,
        cutoff_frequency=args.cutoff_frequency,
        head_animation=args.head_animation,
        animation_clip=args.animation_clip,
        animation_derating=args.animation_derating,
    )
    print("Done instantiating RLWalk")
    rl_walk.run()

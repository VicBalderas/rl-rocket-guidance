import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame


class RocketInterceptEnv(gym.Env):
    """
    Rocket (missile) interception environment v4.

    -----------------------------------------------------------------------
    DESIGN PHILOSOPHY
    -----------------------------------------------------------------------
    The policy's ONLY job is directional guidance — steering the missile
    toward a target. It has no control over speed, thrust magnitude, or
    any flight parameter other than gimbal rate. The missile maintains a
    constant speed at all times, like a bullet with a steerable fin.

    This constraint is enforced as a hard post-physics normalisation:
    after the full gimbal→torque→rotation→thrust physics step, the
    velocity vector is rescaled to a fixed magnitude. The direction comes
    from the realistic physics chain; the magnitude is locked.

    -----------------------------------------------------------------------
    PHYSICS CHAIN  (realistic TVC model)
    -----------------------------------------------------------------------
    1. Policy outputs gimbal rate command  →  nozzle deflects
    2. Nozzle deflection produces torque   →  body rotates
    3. Thrust acts along body axis         →  velocity direction changes
    4. Velocity magnitude normalized →  constant speed enforced

    Crossflow aerodynamic drag resists lateral velocity relative to the
    body axis, causing the missile to weathervane toward its velocity
    direction. This is physically correct for an airframe in atmosphere
    and prevents the body from pointing one way while flying another.

    Gravity applies a directional bias (the missile must actively steer
    against gravitational sag) but cannot change speed because of the
    constant-speed constraint.

    -----------------------------------------------------------------------
    CURRICULUM LEVELS  (target_motion parameter)
    -----------------------------------------------------------------------
    "static"
        Target does not move. One obstacle in a non-threatening location
        (corner/edge) so obstacle observation channels are always live.
        Teaches: basic steering, interception geometry, nose alignment.

    "simple"
        Target moves along a single axis (vertical OR horizontal, chosen
        randomly each episode) at moderate speed. One passive obstacle.
        Teaches: lead pursuit against predictable 1D motion.

    "simple_fast"
        Same as simple but target moves faster.
        Teaches: earlier commitment, tighter pursuit geometry.

    "dual_axis_evasive"
        Target moves in 2D with moderate speed and mild perpendicular
        evasion — when the missile gets close, the target dodges
        perpendicular to the missile's approach vector. 1 passive
        obstacle plus 1 near-target obstacle.
        Teaches: pursuit against targets that dodge, overshoot recovery.

    "evasive"
        Target moves faster with aggressive perpendicular evasion.
        Dodges sharply when missile is close, picking the best
        perpendicular direction based on available arena space.
        2 obstacles near the target region.
        Teaches: committed pursuit against a reactive adversary.

    "evasive_flares"
        Target uses aggressive perpendicular evasion AND tries to
        position obstacles between itself and the missile as shields.
        2-3 obstacles near the target.
        Teaches: avoiding traps while maintaining pursuit.

    "human"
        Target is controlled by keyboard (WASD/arrow keys).
        Only works with render_mode="human". For evaluation only.
        Teaches: nothing (stress-test mode for human evaluation).

    -----------------------------------------------------------------------
    ACTION SPACE
    -----------------------------------------------------------------------
    Box(shape=(1,)) in [-1, 1]
    Controls gimbal rate. Positive deflects nozzle one way, producing
    torque that rotates the body, which curves the flight path.

    -----------------------------------------------------------------------
    OBSERVATION SPACE  (19 floats)
    -----------------------------------------------------------------------
     0   rocket_x / world_width
     1   rocket_y / world_height
     2   cos(velocity_angle)            velocity direction (x component)
     3   sin(velocity_angle)            velocity direction (y component)
     4   cos(body_angle)                body heading (x component)
     5   sin(body_angle)                body heading (y component)
     6   angular_velocity / max_ang_speed
     7   gimbal_angle / max_gimbal_angle
     8   rel_target_x / world_width     target offset x
     9   rel_target_y / world_height    target offset y
    10   target_vx / missile_speed      target velocity x (normalized to missile speed)
    11   target_vy / missile_speed      target velocity y
    12   dist_to_target / max_dist
    13   heading_error / pi             signed angle: nose to LOS
    14   velocity_heading_error / pi    signed angle: velocity dir to LOS
    15   closing_rate / missile_speed   rate of range decrease (positive = closing)
    16   obs_dx / max_dist              nearest obstacle offset x
    17   obs_dy / max_dist              nearest obstacle offset y
    18   obs_edge / max_dist            nearest obstacle edge distance
    """

    metadata = {"render_modes": ["human"], "render_fps": 60}

    # -----------------------------------------------------------------
    # Valid curriculum levels
    # -----------------------------------------------------------------
    VALID_MOTIONS = {
        "static", "simple", "simple_fast",
        "dual_axis_evasive", "evasive", "evasive_flares",
        "human",
    }

    def __init__(
        self,
        target_motion="static",
        render_mode=None,
        max_steps=600,
        world_width=900,
        world_height=650,
        missile_speed=220.0,
        gimbal_limit_deg=20.0,
        gimbal_rate_limit_deg=120.0,
    ):
        super().__init__()

        if target_motion not in self.VALID_MOTIONS:
            raise ValueError(
                f"target_motion must be one of {self.VALID_MOTIONS}, "
                f"got '{target_motion}'"
            )

        if target_motion == "human" and render_mode != "human":
            raise ValueError(
                "target_motion='human' requires render_mode='human'"
            )

        self.target_motion = target_motion
        self.render_mode = render_mode
        self.max_steps = int(max_steps)

        self.world_width = int(world_width)
        self.world_height = int(world_height)
        self.max_dist = float(np.hypot(self.world_width, self.world_height))

        # ----- Missile physics -----
        self.dt = 0.05
        self.missile_speed = float(missile_speed)       # CONSTANT magnitude
        self.main_thrust_power = 28.0                   # only sets torque authority
        self.rocket_mass = 1.0
        self.angular_damping = 0.92
        self.max_ang_speed = 5.0
        self.crossflow_damping = 0.82                   # strong weathervane effect
        self.gravity_bias = 1.5                          # directional sag only

        # Gimbal dynamics
        self.max_gimbal_angle = np.deg2rad(gimbal_limit_deg)
        self.max_gimbal_rate = np.deg2rad(gimbal_rate_limit_deg)
        self.gimbal_torque_coeff = 22.0

        # Geometry / collision
        self.rocket_length = 28
        self.rocket_width = 12
        self.rocket_collision_radius = 12.0
        self.intercept_radius = 22.0       # proximity fuze trigger distance
        self.obstacle_radius = 26.0

        # Target speed definitions per curriculum level
        self.target_speeds = {
            "static":             0.0,
            "simple":            55.0,
            "simple_fast":       90.0,
            "dual_axis_evasive": 85.0,
            "evasive":          100.0,
            "evasive_flares":   100.0,
            "human":            120.0,    # max human-controlled speed
        }

        # Evasion parameters per level
        # dodge_radius: distance at which target starts dodging
        # dodge_force: how hard the perpendicular dodge is
        self.evasion_params = {
            "dual_axis_evasive": {"dodge_radius": 200.0, "dodge_force": 180.0},
            "evasive":           {"dodge_radius": 280.0, "dodge_force": 300.0},
            "evasive_flares":    {"dodge_radius": 280.0, "dodge_force": 300.0},
        }

        # ----- Spaces -----
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(19,), dtype=np.float32,
        )

        # ----- Render -----
        self.screen = None
        self.clock = None

        # ----- State (populated in reset) -----
        self.rocket_x = 0.0
        self.rocket_y = 0.0
        self.rocket_vx = 0.0
        self.rocket_vy = 0.0
        self.rocket_angle = 0.0            # body heading
        self.rocket_angular_velocity = 0.0
        self.nozzle_gimbal_angle = 0.0

        self.target_x = 0.0
        self.target_y = 0.0
        self.target_vx = 0.0
        self.target_vy = 0.0
        self.target_axis = "vertical"      # for simple/simple_fast modes

        # Dodge state
        self.dodge_dir = None
        self.dodge_timer = 0

        # Human control state
        self.human_keys = {"up": False, "down": False, "left": False, "right": False}

        self.obstacles = []
        self.steps = 0
        self.prev_dist_to_target = 0.0

    # =================================================================
    #  Utility
    # =================================================================
    @staticmethod
    def _wrap_angle(angle):
        return np.arctan2(np.sin(angle), np.cos(angle))

    # =================================================================
    #  Spawning
    # =================================================================
    def _spawn_missile(self):
        """Missile starts left-center, pointing right, at full speed."""
        self.rocket_x = self.world_width * 0.08
        self.rocket_y = self.world_height * 0.5

        self.rocket_angle = 0.0                          # pointing right
        self.rocket_vx = self.missile_speed               # full speed rightward
        self.rocket_vy = 0.0
        self.rocket_angular_velocity = 0.0
        self.nozzle_gimbal_angle = 0.0

    def _spawn_target(self):
        """
        Spawn target in the right portion of the arena with a buffer
        zone from the missile so it has room to maneuver.
        """
        # Horizontal: right 40% of arena (x ∈ [0.55, 0.92])
        self.target_x = float(self.np_random.uniform(
            self.world_width * 0.55,
            self.world_width * 0.92,
        ))
        # Vertical: most of the arena height with margin
        self.target_y = float(self.np_random.uniform(
            self.world_height * 0.10,
            self.world_height * 0.90,
        ))

        speed = self.target_speeds[self.target_motion]

        if self.target_motion == "static":
            self.target_vx = 0.0
            self.target_vy = 0.0

        elif self.target_motion in ("simple", "simple_fast"):
            # Single-axis movement, randomly vertical or horizontal.
            self.target_axis = self.np_random.choice(["vertical", "horizontal"])
            direction = self.np_random.choice([-1.0, 1.0])
            if self.target_axis == "vertical":
                self.target_vx = 0.0
                self.target_vy = float(speed * direction)
            else:
                self.target_vx = float(speed * direction)
                self.target_vy = 0.0

        elif self.target_motion in ("dual_axis_evasive", "evasive", "evasive_flares"):
            # Initial 2D velocity — evasion logic takes over each step.
            angle = float(self.np_random.uniform(0, 2.0 * np.pi))
            self.target_vx = float(speed * 0.5 * np.cos(angle))
            self.target_vy = float(speed * 0.5 * np.sin(angle))

        elif self.target_motion == "human":
            self.target_vx = 0.0
            self.target_vy = 0.0

    def _spawn_obstacles(self):
        """
        Spawn obstacles according to curriculum level.
        Early levels: 1 obstacle in a non-threatening corner/edge.
        Later levels: obstacles near the target region.
        """
        self.obstacles = []

        if self.target_motion in ("static", "simple", "simple_fast"):
            # One passive obstacle placed along an edge or corner,
            # away from the likely intercept corridor.
            self.obstacles.append(self._edge_obstacle())

        elif self.target_motion == "dual_axis_evasive":
            # 1 edge obstacle + 1 near-target obstacle.
            self.obstacles.append(self._edge_obstacle())
            self.obstacles.append(self._target_region_obstacle())

        elif self.target_motion == "evasive":
            # 2 obstacles near the target region.
            for _ in range(2):
                self.obstacles.append(self._target_region_obstacle())

        elif self.target_motion in ("evasive_flares", "human"):
            # 2-3 obstacles near the target — these are the "flares".
            n = int(self.np_random.integers(2, 4))  # 2 or 3
            for _ in range(n):
                self.obstacles.append(self._target_region_obstacle())

    def _edge_obstacle(self):
        """
        Place an obstacle along an arena edge or corner, well away from
        the intercept corridor (missile starts left-center, target is right).
        """
        # Pick a random edge: top, bottom, top-right corner, bottom-right corner.
        choice = int(self.np_random.integers(0, 4))
        margin = self.obstacle_radius + 20.0

        if choice == 0:  # top edge
            ox = float(self.np_random.uniform(self.world_width * 0.3, self.world_width * 0.7))
            oy = margin
        elif choice == 1:  # bottom edge
            ox = float(self.np_random.uniform(self.world_width * 0.3, self.world_width * 0.7))
            oy = self.world_height - margin
        elif choice == 2:  # top-right corner
            ox = float(self.np_random.uniform(self.world_width * 0.80, self.world_width * 0.95))
            oy = float(self.np_random.uniform(margin, self.world_height * 0.25))
        else:  # bottom-right corner
            ox = float(self.np_random.uniform(self.world_width * 0.80, self.world_width * 0.95))
            oy = float(self.np_random.uniform(self.world_height * 0.75, self.world_height - margin))

        return {"x": ox, "y": oy, "r": self.obstacle_radius}

    def _target_region_obstacle(self):
        """
        Place an obstacle in the target's region. Ensures minimum distance
        from the missile spawn and from other obstacles.
        """
        margin = self.obstacle_radius + 15.0
        for _ in range(200):
            # Spawn within a radius of the target's initial position.
            ox = self.target_x + float(self.np_random.uniform(-150.0, 150.0))
            oy = self.target_y + float(self.np_random.uniform(-150.0, 150.0))

            # Clamp inside arena with margin.
            ox = float(np.clip(ox, margin, self.world_width - margin))
            oy = float(np.clip(oy, margin, self.world_height - margin))

            # Must be away from missile spawn.
            d_missile = np.hypot(ox - self.rocket_x, oy - self.rocket_y)
            if d_missile < 180.0:
                continue

            # Must be away from target (don't block it immediately).
            d_target = np.hypot(ox - self.target_x, oy - self.target_y)
            if d_target < 60.0:
                continue

            # Must be away from existing obstacles.
            too_close = False
            for obs in self.obstacles:
                if np.hypot(ox - obs["x"], oy - obs["y"]) < 70.0:
                    too_close = True
                    break
            if too_close:
                continue

            return {"x": ox, "y": oy, "r": self.obstacle_radius}

        # Fallback if we can't find a valid spot.
        return {
            "x": float(np.clip(self.target_x + 100.0, margin, self.world_width - margin)),
            "y": float(np.clip(self.target_y + 100.0, margin, self.world_height - margin)),
            "r": self.obstacle_radius,
        }

    # =================================================================
    #  Target movement
    # =================================================================
    def _move_target(self):
        if self.target_motion == "static":
            return

        if self.target_motion == "human":
            self._human_target_movement()
            return

        speed_cap = self.target_speeds[self.target_motion] * 1.2
        margin = 60.0

        if self.target_motion in ("simple", "simple_fast"):
            # Single-axis bounce.
            self.target_x += self.target_vx * self.dt
            self.target_y += self.target_vy * self.dt

            if self.target_axis == "vertical":
                if self.target_y < margin:
                    self.target_y = margin
                    self.target_vy = abs(self.target_vy)
                elif self.target_y > self.world_height - margin:
                    self.target_y = self.world_height - margin
                    self.target_vy = -abs(self.target_vy)
            else:
                if self.target_x < self.world_width * 0.35:
                    self.target_x = self.world_width * 0.35
                    self.target_vx = abs(self.target_vx)
                elif self.target_x > self.world_width - margin:
                    self.target_x = self.world_width - margin
                    self.target_vx = -abs(self.target_vx)

        elif self.target_motion in ("dual_axis_evasive", "evasive", "evasive_flares"):
            self._perpendicular_evasive_movement(speed_cap)

        # Speed cap.
        speed = np.hypot(self.target_vx, self.target_vy)
        if speed > speed_cap:
            self.target_vx *= speed_cap / speed
            self.target_vy *= speed_cap / speed

    def _perpendicular_evasive_movement(self, speed_cap):
        margin = 60.0
        params = self.evasion_params.get(
            self.target_motion,
            {"dodge_radius": 200.0, "dodge_force": 180.0},
        )
        dodge_radius = params["dodge_radius"]

        missile_speed = np.hypot(self.rocket_vx, self.rocket_vy)
        dist_to_missile = np.hypot(
            self.rocket_x - self.target_x,
            self.rocket_y - self.target_y,
        )

        # 🔁 RESET dodge when missile is far
        if dist_to_missile >= dodge_radius * 1.6:
            self.dodge_timer = 0
            self.dodge_dir = None

        # 🔥 EARLIER TRIGGER
        if missile_speed > 1e-6 and dist_to_missile < dodge_radius * 1.6:

            approach_x = self.rocket_vx / missile_speed
            approach_y = self.rocket_vy / missile_speed

            perp_a_x = -approach_y
            perp_a_y = approach_x
            perp_b_x = approach_y
            perp_b_y = -approach_x

            future_a_x = self.target_x + perp_a_x * 100.0
            future_a_y = self.target_y + perp_a_y * 100.0
            future_b_x = self.target_x + perp_b_x * 100.0
            future_b_y = self.target_y + perp_b_y * 100.0

            score_a = min(
                future_a_x - margin,
                self.world_width - margin - future_a_x,
                future_a_y - margin,
                self.world_height - margin - future_a_y,
            )
            score_b = min(
                future_b_x - margin,
                self.world_width - margin - future_b_x,
                future_b_y - margin,
                self.world_height - margin - future_b_y,
            )

            # 🔥 COMMITMENT (prevents flip)
            if self.dodge_timer <= 0:
                if score_a >= score_b:
                    self.dodge_dir = (perp_a_x, perp_a_y)
                else:
                    self.dodge_dir = (perp_b_x, perp_b_y)
                self.dodge_timer = 16

            dodge_x, dodge_y = self.dodge_dir
            self.dodge_timer -= 1

            # 🎯 PROXIMITY-BASED STRENGTH
            proximity = 1.0 - np.clip(dist_to_missile / dodge_radius, 0.0, 1.0)

            # Never too weak, but much stronger when close
            strength = 0.4 + 0.6 * (proximity ** 2)

            desired_vx = dodge_x * speed_cap * strength
            desired_vy = dodge_y * speed_cap * strength

            blend = 0.8
            self.target_vx = (1 - blend) * self.target_vx + blend * desired_vx
            self.target_vy = (1 - blend) * self.target_vy + blend * desired_vy

        # Flare logic
        if self.target_motion == "evasive_flares" and self.obstacles:
            self._flare_luring()

        # Obstacle avoidance
        for obs in self.obstacles:
            d = np.hypot(obs["x"] - self.target_x, obs["y"] - self.target_y)
            if 100.0 > d > 1e-6:
                push_x = (self.target_x - obs["x"]) / d
                push_y = (self.target_y - obs["y"]) / d
                proximity = 1.0 - np.clip(d / 100.0, 0.0, 1.0)
                push_force = 120.0 * proximity
                self.target_vx += push_x * push_force * self.dt
                self.target_vy += push_y * push_force * self.dt

        # 🔥 REDUCED NOISE
        noise_scale = 2.0 if self.target_motion == "dual_axis_evasive" else 1.0
        self.target_vx += float(self.np_random.uniform(-noise_scale, noise_scale))
        self.target_vy += float(self.np_random.uniform(-noise_scale, noise_scale))

        # Integrate
        self.target_x += self.target_vx * self.dt
        self.target_y += self.target_vy * self.dt

        # Bounds
        if self.target_x < margin:
            self.target_x = margin
            self.target_vx = abs(self.target_vx)
        elif self.target_x > self.world_width - margin:
            self.target_x = self.world_width - margin
            self.target_vx = -abs(self.target_vx)

        if self.target_y < margin:
            self.target_y = margin
            self.target_vy = abs(self.target_vy)
        elif self.target_y > self.world_height - margin:
            self.target_y = self.world_height - margin
            self.target_vy = -abs(self.target_vy)

    def _flare_luring(self):
        """
        In evasive_flares mode, the target tries to position an obstacle
        between itself and the missile as a shield.
        """
        best_lure_score = -1.0
        best_lure_obs = None

        for obs in self.obstacles:
            to_obs_x = obs["x"] - self.target_x
            to_obs_y = obs["y"] - self.target_y
            to_obs_dist = np.hypot(to_obs_x, to_obs_y)

            to_missile_x = self.rocket_x - self.target_x
            to_missile_y = self.rocket_y - self.target_y
            to_missile_dist = np.hypot(to_missile_x, to_missile_y)

            if to_obs_dist < 1e-6 or to_missile_dist < 1e-6:
                continue

            alignment = (
                (to_obs_x * to_missile_x + to_obs_y * to_missile_y)
                / (to_obs_dist * to_missile_dist)
            )

            if alignment > 0.2 and to_obs_dist < 300.0:
                score = alignment / (to_obs_dist / 100.0 + 0.5)
                if score > best_lure_score:
                    best_lure_score = score
                    best_lure_obs = obs

        if best_lure_obs is not None:
            obs_x = best_lure_obs["x"]
            obs_y = best_lure_obs["y"]

            om_x = self.rocket_x - obs_x
            om_y = self.rocket_y - obs_y
            om_dist = np.hypot(om_x, om_y)

            if om_dist > 1e-6:
                hide_x = obs_x - (om_x / om_dist) * (best_lure_obs["r"] + 40.0)
                hide_y = obs_y - (om_y / om_dist) * (best_lure_obs["r"] + 40.0)

                lure_dx = hide_x - self.target_x
                lure_dy = hide_y - self.target_y
                lure_dist = np.hypot(lure_dx, lure_dy)

                if lure_dist > 1e-6:
                    lure_force = 100.0
                    self.target_vx += (lure_dx / lure_dist) * lure_force * self.dt
                    self.target_vy += (lure_dy / lure_dist) * lure_force * self.dt

    def _human_target_movement(self):
        """
        Human controls the target with keyboard.
        WASD or arrow keys set velocity directly.
        """
        speed = self.target_speeds["human"]
        accel = 300.0  # acceleration for responsive control
        friction = 0.92  # friction so target slows when keys released

        if self.human_keys["up"]:
            self.target_vy -= accel * self.dt
        if self.human_keys["down"]:
            self.target_vy += accel * self.dt
        if self.human_keys["left"]:
            self.target_vx -= accel * self.dt
        if self.human_keys["right"]:
            self.target_vx += accel * self.dt

        # Apply friction when no keys pressed.
        any_key = any(self.human_keys.values())
        if not any_key:
            self.target_vx *= friction
            self.target_vy *= friction

        # Speed cap.
        s = np.hypot(self.target_vx, self.target_vy)
        if s > speed:
            self.target_vx *= speed / s
            self.target_vy *= speed / s

        # Integrate.
        self.target_x += self.target_vx * self.dt
        self.target_y += self.target_vy * self.dt

        # Boundary clamp.
        margin = 30.0
        if self.target_x < margin:
            self.target_x = margin
            self.target_vx = 0.0
        elif self.target_x > self.world_width - margin:
            self.target_x = self.world_width - margin
            self.target_vx = 0.0
        if self.target_y < margin:
            self.target_y = margin
            self.target_vy = 0.0
        elif self.target_y > self.world_height - margin:
            self.target_y = self.world_height - margin
            self.target_vy = 0.0

    # =================================================================
    #  Observation
    # =================================================================
    def _closest_obstacle_features(self):
        if not self.obstacles:
            return 0.0, 0.0, 1.0

        best_center_dist = np.inf
        best_dx = 0.0
        best_dy = 0.0
        best_edge_dist = self.max_dist

        for obs in self.obstacles:
            dx = obs["x"] - self.rocket_x
            dy = obs["y"] - self.rocket_y
            center_dist = np.hypot(dx, dy)
            if center_dist < best_center_dist:
                best_center_dist = center_dist
                best_dx = dx
                best_dy = dy
                best_edge_dist = center_dist - (obs["r"] + self.rocket_collision_radius)

        return (
            best_dx / self.max_dist,
            best_dy / self.max_dist,
            np.clip(best_edge_dist / self.max_dist, -1.0, 1.0),
        )

    def _get_observation(self):
        # Velocity angle (direction the missile is actually moving).
        vel_angle = np.arctan2(self.rocket_vy, self.rocket_vx)

        # Relative target state.
        rel_tx = self.target_x - self.rocket_x
        rel_ty = self.target_y - self.rocket_y
        dist_to_target = np.hypot(rel_tx, rel_ty)

        # Line of sight angle.
        los_angle = np.arctan2(rel_ty, rel_tx)

        # Heading error: body nose to LOS (signed, normalized).
        heading_error = self._wrap_angle(los_angle - self.rocket_angle)

        # Velocity heading error: velocity direction to LOS (signed, normalized).
        vel_heading_error = self._wrap_angle(los_angle - vel_angle)

        # Closing rate: rate of range decrease (positive = closing).
        rel_vx = self.target_vx - self.rocket_vx
        rel_vy = self.target_vy - self.rocket_vy
        if dist_to_target > 1e-6:
            los_ux = rel_tx / dist_to_target
            los_uy = rel_ty / dist_to_target
            # Negative because closing means relative velocity is toward target.
            closing_rate = -(rel_vx * los_ux + rel_vy * los_uy)
        else:
            closing_rate = 0.0

        obs_dx, obs_dy, obs_edge = self._closest_obstacle_features()

        return np.array([
            self.rocket_x / self.world_width,                    # 0
            self.rocket_y / self.world_height,                   # 1
            np.cos(vel_angle),                                   # 2
            np.sin(vel_angle),                                   # 3
            np.cos(self.rocket_angle),                           # 4
            np.sin(self.rocket_angle),                           # 5
            self.rocket_angular_velocity / self.max_ang_speed,   # 6
            self.nozzle_gimbal_angle / self.max_gimbal_angle,    # 7
            rel_tx / self.world_width,                           # 8
            rel_ty / self.world_height,                          # 9
            self.target_vx / self.missile_speed,                 # 10
            self.target_vy / self.missile_speed,                 # 11
            dist_to_target / self.max_dist,                      # 12
            heading_error / np.pi,                               # 13
            vel_heading_error / np.pi,                           # 14
            closing_rate / self.missile_speed,                   # 15
            obs_dx,                                              # 16
            obs_dy,                                              # 17
            obs_edge,                                            # 18
        ], dtype=np.float32)

    # =================================================================
    #  Step
    # =================================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.human_keys = {"up": False, "down": False, "left": False, "right": False}

        self._spawn_missile()
        self._spawn_target()
        self._spawn_obstacles()

        self.prev_dist_to_target = float(np.hypot(
            self.target_x - self.rocket_x,
            self.target_y - self.rocket_y,
        ))

        return self._get_observation(), {}

    def step(self, action):
        self.steps += 1

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        gimbal_rate_cmd = float(np.clip(action[0], -1.0, 1.0))

        # ---------------------------------------------------------------
        # 1. Gimbal dynamics
        # ---------------------------------------------------------------
        self.nozzle_gimbal_angle += gimbal_rate_cmd * self.max_gimbal_rate * self.dt
        self.nozzle_gimbal_angle = float(np.clip(
            self.nozzle_gimbal_angle,
            -self.max_gimbal_angle,
            self.max_gimbal_angle,
        ))

        # ---------------------------------------------------------------
        # 2. Torque from gimbal → body rotation
        # ---------------------------------------------------------------
        torque = (
            self.gimbal_torque_coeff
            * np.sin(self.nozzle_gimbal_angle)
        )
        self.rocket_angular_velocity += torque * self.dt
        self.rocket_angular_velocity *= self.angular_damping
        self.rocket_angular_velocity = float(np.clip(
            self.rocket_angular_velocity,
            -self.max_ang_speed,
            self.max_ang_speed,
        ))
        self.rocket_angle = self._wrap_angle(
            self.rocket_angle + self.rocket_angular_velocity * self.dt
        )

        # ---------------------------------------------------------------
        # 3. Thrust along body axis → velocity update
        # ---------------------------------------------------------------
        body_ax = np.cos(self.rocket_angle)
        body_ay = np.sin(self.rocket_angle)

        # Thrust acceleration along body axis.
        thrust_ax = (self.main_thrust_power / self.rocket_mass) * body_ax
        thrust_ay = (self.main_thrust_power / self.rocket_mass) * body_ay

        self.rocket_vx += thrust_ax * self.dt
        self.rocket_vy += thrust_ay * self.dt

        # Gravity bias: pulls velocity downward (screen y increases down).
        self.rocket_vy += self.gravity_bias * self.dt

        # ---------------------------------------------------------------
        # 4. Crossflow drag: resist lateral velocity relative to body axis.
        #    This makes the missile weathervane toward its velocity.
        # ---------------------------------------------------------------
        lateral = np.array([-body_ay, body_ax], dtype=np.float64)
        vel = np.array([self.rocket_vx, self.rocket_vy], dtype=np.float64)
        crossflow_speed = float(np.dot(vel, lateral))

        self.rocket_vx -= crossflow_speed * (1.0 - self.crossflow_damping) * lateral[0]
        self.rocket_vy -= crossflow_speed * (1.0 - self.crossflow_damping) * lateral[1]

        # ---------------------------------------------------------------
        # 5. CONSTANT SPEED ENFORCEMENT
        #    Direction from physics. Magnitude locked.
        # ---------------------------------------------------------------
        current_speed = np.hypot(self.rocket_vx, self.rocket_vy)
        if current_speed > 1e-8:
            scale = self.missile_speed / current_speed
            self.rocket_vx *= scale
            self.rocket_vy *= scale

        # ---------------------------------------------------------------
        # 6. Position update
        # ---------------------------------------------------------------
        self.rocket_x += self.rocket_vx * self.dt
        self.rocket_y += self.rocket_vy * self.dt

        # ---------------------------------------------------------------
        # 7. Move target
        # ---------------------------------------------------------------
        self._move_target()

        # ---------------------------------------------------------------
        # 8. Collision detection
        # ---------------------------------------------------------------
        dist_to_target = float(np.hypot(
            self.target_x - self.rocket_x,
            self.target_y - self.rocket_y,
        ))

        # Proximity fuze: intercept if close enough.
        hit_target = dist_to_target <= self.intercept_radius

        # Obstacle collision (body collision radius).
        hit_obstacle = False
        min_obstacle_edge = np.inf
        for obs in self.obstacles:
            center_dist = float(np.hypot(
                obs["x"] - self.rocket_x,
                obs["y"] - self.rocket_y,
            ))
            edge_dist = center_dist - (obs["r"] + self.rocket_collision_radius)
            min_obstacle_edge = min(min_obstacle_edge, edge_dist)
            if center_dist <= (obs["r"] + self.rocket_collision_radius):
                hit_obstacle = True
                break

        out_of_bounds = (
            self.rocket_x < 0.0 or self.rocket_x > self.world_width
            or self.rocket_y < 0.0 or self.rocket_y > self.world_height
        )
        timeout = self.steps >= self.max_steps

        terminated = bool(hit_target or hit_obstacle or out_of_bounds)
        truncated = bool(timeout)

        # ---------------------------------------------------------------
        # 9. Reward
        # ---------------------------------------------------------------
        reward = self._compute_reward(
            dist_to_target, min_obstacle_edge,
            hit_target, hit_obstacle, out_of_bounds, timeout,
        )

        observation = self._get_observation()
        info = {
            "distance_to_target": dist_to_target,
            "hit_target": hit_target,
            "hit_obstacle": hit_obstacle,
            "out_of_bounds": out_of_bounds,
            "timeout": timeout,
            "target_motion": self.target_motion,
            "steps": self.steps,
        }

        return observation, float(reward), terminated, truncated, info

    # =================================================================
    #  Reward  (focused on interception geometry)
    # =================================================================
    def _compute_reward(
        self, dist_to_target, min_obstacle_edge,
        hit_target, hit_obstacle, out_of_bounds, timeout,
    ):
        reward = 0.0

        # --- Primary signal: close distance to target ---
        progress = (self.prev_dist_to_target - dist_to_target) / self.max_dist
        self.prev_dist_to_target = dist_to_target
        reward += 10.0 * progress

        # --- Heading alignment: nose pointed at target ---
        rel_tx = self.target_x - self.rocket_x
        rel_ty = self.target_y - self.rocket_y
        los_angle = np.arctan2(rel_ty, rel_tx)
        heading_error = abs(self._wrap_angle(los_angle - self.rocket_angle)) / np.pi
        reward += 0.04 * (1.0 - heading_error)

        # --- Small living cost for urgency ---
        reward -= 0.005

        # --- Obstacle proximity warning ---
        if np.isfinite(min_obstacle_edge) and min_obstacle_edge < 80.0:
            closeness = 1.0 - np.clip(min_obstacle_edge / 80.0, 0.0, 1.0)
            reward -= 0.20 * closeness

        # --- Terminal rewards ---
        if hit_target:
            reward += 15.0
        elif hit_obstacle:
            reward -= 8.0
        elif out_of_bounds:
            reward -= 5.0
        elif timeout:
            reward -= 3.0

        return reward

    # =================================================================
    #  Rendering
    # =================================================================
    def _draw_rocket(self):
        body = np.array([
            [self.rocket_length * 0.55, 0.0],
            [-self.rocket_length * 0.45, -self.rocket_width * 0.55],
            [-self.rocket_length * 0.25, 0.0],
            [-self.rocket_length * 0.45, self.rocket_width * 0.55],
        ], dtype=np.float32)

        c = np.cos(self.rocket_angle)
        s = np.sin(self.rocket_angle)
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)
        pts = body @ rot.T
        pts[:, 0] += self.rocket_x
        pts[:, 1] += self.rocket_y
        pygame.draw.polygon(
            self.screen, (220, 220, 230),
            [(int(x), int(y)) for x, y in pts],
        )

        # Cockpit window.
        window_local = np.array([4.0, 0.0], dtype=np.float32) @ rot.T
        wx = int(self.rocket_x + window_local[0])
        wy = int(self.rocket_y + window_local[1])
        pygame.draw.circle(self.screen, (130, 200, 255), (wx, wy), 4)

        # Thrust flame (always on — constant thrust).
        nozzle_angle = self.rocket_angle + self.nozzle_gimbal_angle
        back = np.array([-self.rocket_length * 0.52, 0.0], dtype=np.float32) @ rot.T
        bx = self.rocket_x + back[0]
        by = self.rocket_y + back[1]

        flame_dir = np.array([
            -np.cos(nozzle_angle),
            -np.sin(nozzle_angle),
        ], dtype=np.float32)
        flame_tip = np.array([bx, by], dtype=np.float32) + flame_dir * 20.0
        side = np.array([-flame_dir[1], flame_dir[0]], dtype=np.float32) * 4.5

        flame_pts = np.array([
            [bx + side[0], by + side[1]],
            [flame_tip[0], flame_tip[1]],
            [bx - side[0], by - side[1]],
        ], dtype=np.float32)
        pygame.draw.polygon(
            self.screen, (255, 170, 60),
            [(int(x), int(y)) for x, y in flame_pts],
        )

        # Nozzle direction line.
        nozzle_end = np.array([bx, by], dtype=np.float32) + np.array([
            -np.cos(nozzle_angle),
            -np.sin(nozzle_angle),
        ], dtype=np.float32) * 12.0
        pygame.draw.line(
            self.screen, (255, 240, 120),
            (int(bx), int(by)),
            (int(nozzle_end[0]), int(nozzle_end[1])),
            2,
        )

    def render(self):
        if self.render_mode != "human":
            return

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(
                (self.world_width, self.world_height)
            )
            pygame.display.set_caption("Missile Intercept v4")
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return
            # Human target control — key press/release tracking.
            if self.target_motion == "human":
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_w, pygame.K_UP):
                        self.human_keys["up"] = True
                    elif event.key in (pygame.K_s, pygame.K_DOWN):
                        self.human_keys["down"] = True
                    elif event.key in (pygame.K_a, pygame.K_LEFT):
                        self.human_keys["left"] = True
                    elif event.key in (pygame.K_d, pygame.K_RIGHT):
                        self.human_keys["right"] = True
                elif event.type == pygame.KEYUP:
                    if event.key in (pygame.K_w, pygame.K_UP):
                        self.human_keys["up"] = False
                    elif event.key in (pygame.K_s, pygame.K_DOWN):
                        self.human_keys["down"] = False
                    elif event.key in (pygame.K_a, pygame.K_LEFT):
                        self.human_keys["left"] = False
                    elif event.key in (pygame.K_d, pygame.K_RIGHT):
                        self.human_keys["right"] = False

        self.screen.fill((10, 12, 28))

        # Arena border.
        pygame.draw.rect(
            self.screen, (50, 55, 85),
            (0, 0, self.world_width, self.world_height), 2,
        )

        # Obstacles.
        for obs in self.obstacles:
            pygame.draw.circle(
                self.screen, (175, 85, 50),
                (int(obs["x"]), int(obs["y"])), int(obs["r"]),
            )
            pygame.draw.circle(
                self.screen, (225, 130, 85),
                (int(obs["x"]), int(obs["y"])), int(max(obs["r"] - 7, 4)),
            )

        # Target.
        target_color = (50, 235, 90) if self.target_motion != "human" else (90, 180, 255)
        pygame.draw.circle(
            self.screen, target_color,
            (int(self.target_x), int(self.target_y)),
            int(self.intercept_radius),
        )
        pygame.draw.circle(
            self.screen, (20, 150, 45) if self.target_motion != "human" else (40, 100, 200),
            (int(self.target_x), int(self.target_y)), 5,
        )

        # Line of sight.
        pygame.draw.line(
            self.screen, (70, 120, 220),
            (int(self.rocket_x), int(self.rocket_y)),
            (int(self.target_x), int(self.target_y)),
            1,
        )

        self._draw_rocket()

        # HUD.
        font = pygame.font.SysFont("consoles", 15)
        distance = np.hypot(
            self.target_x - self.rocket_x,
            self.target_y - self.rocket_y,
        )
        speed = np.hypot(self.rocket_vx, self.rocket_vy)
        hud_lines = [
            f"MODE   {self.target_motion.upper()}",
            f"STEP   {self.steps}/{self.max_steps}",
            f"DIST   {distance:7.1f}",
            f"SPEED  {speed:7.1f}",
            f"ANGLE  {np.degrees(self.rocket_angle):7.1f} deg",
            f"GIMBAL {np.degrees(self.nozzle_gimbal_angle):7.1f} deg",
        ]
        if self.target_motion == "human":
            hud_lines.append("CTRL   WASD / ARROWS")

        x0, y0 = 18, 16
        for i, line in enumerate(hud_lines):
            surf = font.render(line, True, (210, 210, 215))
            self.screen.blit(surf, (x0, y0 + i * 20))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None


# =====================================================================
#  Quick test
# =====================================================================
if __name__ == "__main__":
    env = RocketInterceptEnv(
        render_mode="human",
        target_motion="static",
    )
    obs, _ = env.reset(seed=42)
    print("Observation shape:", obs.shape)
    print("Action space:", env.action_space)
    print("First observation:", obs)

    for _ in range(800):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            print("Episode ended:", info)
            obs, _ = env.reset()

    env.close()
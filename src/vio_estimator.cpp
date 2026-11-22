// =====================================================================================
// vio_estimator.cpp — sliding-window keypoint visual-inertial odometry (VIO).
//
// This is a standalone port of Basalt's sqrt_keypoint_vio estimator. The overall
// pipeline executed for every new camera frame (see VioEstimator::measure) is:
//
//   1. IMU prediction   : propagate the newest full state with the preintegrated IMU
//                         measurement and add it to the window.
//   2. Data association : attach the new frame's optical-flow keypoints to landmarks
//                         already in the map.
//   3. Keyframe / map   : if too few keypoints are tracked, make this frame a keyframe
//                         and triangulate new landmarks hosted in it.
//   4. Optimization     : Levenberg–Marquardt over all window states and landmarks,
//                         with landmarks eliminated by the Schur complement.
//   5. Marginalization  : drop old states / keyframes / lost landmarks from the window
//                         and fold their information into a Gaussian prior.
//
// ---------------------------------------------------------------------------------
// STATE PARAMETRIZATION AND POSE UPDATES
// ---------------------------------------------------------------------------------
// The window holds two kinds of frame variables, both keyed by timestamp (ns):
//
//   * frame_states_ : full NavState, 15-dof increment (POSE_VEL_BIAS_SIZE)
//                       inc = [ dp(0:3) | dphi(3:6) | dv(6:9) | dbg(9:12) | dba(12:15) ]
//   * frame_poses_  : pose-only keyframes, 6-dof increment (POSE_SIZE)
//                       inc = [ dp(0:3) | dphi(3:6) ]
//
// Increments are applied (NavState::applyInc / applyPoseInc in imu.h) as:
//
//   t_w_i <- t_w_i + dp                    (position: additive, world frame)
//   R_w_i <- Exp(dphi) * R_w_i             (rotation: LEFT perturbation, i.e. the
//                                           rotation error is expressed in the WORLD frame)
//   v_w_i <- v_w_i + dv,  bg <- bg + dbg,  ba <- ba + dba   (additive)
//
// All Jacobians in this file are derived consistently with this left/world-frame
// perturbation. The inverse operation ("x [-] x_lin") used by the prior is in
// priorDelta(): dp = t - t_lin, dphi = Log(R * R_lin^T).
//
// Landmarks are parametrized relative to a HOST keyframe/camera as
//   dir      : 2-D stereographic projection of the unit bearing in the host camera
//   inv_dist : inverse distance rho = 1/||p_c|| along that bearing
// and are updated additively (dir += d_dir, inv_dist += d_rho).
//
// ---------------------------------------------------------------------------------
// LINEAR SYSTEM CONVENTION
// ---------------------------------------------------------------------------------
// Every factor with residual r(x), Jacobian J and information W contributes
//     H += J^T W J        (Gauss–Newton Hessian approximation)
//     b += J^T W r        (gradient of 0.5 r^T W r)
// and the Gauss–Newton/LM step solves  H dx = -b  (dx = -H^{-1} b).
// =====================================================================================

#include "bw/vio_estimator.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

namespace bw {

namespace {

using Mat23 = Eigen::Matrix<double, 2, 3>;
using Mat26 = Eigen::Matrix<double, 2, 6>;
using Mat63 = Eigen::Matrix<double, 6, 3>;
using Mat66 = Eigen::Matrix<double, 6, 6>;

// Basalt triangulation (DLT); returns [bearing in frame 0; inverse distance]
// -------------------------------------------------------------------------------
// Two-view linear (DLT) triangulation of a point from two bearing vectors.
//
// Inputs:
//   f0     : bearing vector (not necessarily unit) of the point in camera 0.
//   f1     : bearing vector of the same point in camera 1.
//   T_0_1  : pose of camera 1 expressed in camera 0 (maps points from cam 1 to cam 0).
//
// Output (return value):
//   Vec4 v = [b; rho], where b (3x1) is the UNIT bearing of the point in camera 0 and
//   rho = 1/distance is its inverse distance from camera 0's center, i.e. the
//   homogeneous point is scaled such that ||b|| = 1. If the solution is degenerate
//   (e.g. no parallax) rho can be ~0, negative, or non-finite; the caller checks this.
//
// Method: the projection matrices are P1 = [I | 0] (camera 0) and P2 = [R_1_0 | t_1_0]
// (camera 1). Each view gives two linear equations of the form x*P.row(2) - z*P.row(0)
// (cross-product constraint f x (P X) = 0). The homogeneous solution X is the right
// singular vector of A with the smallest singular value. The sign is fixed so that the
// point lies in front of camera 0 (b . f0 > 0).
// -------------------------------------------------------------------------------
Vec4 triangulate(const Vec3& f0, const Vec3& f1, const SE3& T_0_1) {
  Eigen::Matrix<double, 3, 4> P1 = Eigen::Matrix<double, 3, 4>::Zero();
  P1.leftCols<3>().setIdentity();
  const SE3 T_1_0 = T_0_1.inverse();
  Eigen::Matrix<double, 3, 4> P2;
  P2.leftCols<3>() = T_1_0.R;
  P2.col(3) = T_1_0.t;
  Mat4 A;
  A.row(0) = f0[0] * P1.row(2) - f0[2] * P1.row(0);
  A.row(1) = f0[1] * P1.row(2) - f0[2] * P1.row(1);
  A.row(2) = f1[0] * P2.row(2) - f1[2] * P2.row(0);
  A.row(3) = f1[1] * P2.row(2) - f1[2] * P2.row(1);
  Eigen::JacobiSVD<Mat4> svd(A, Eigen::ComputeFullV);
  Vec4 v = svd.matrixV().col(3);
  // make the point lie in front of camera 0
  if (v.head<3>().dot(f0) < 0) v = -v;
  // normalize so that the first three components are a unit bearing; the 4th
  // component then equals the inverse distance
  return v / v.head<3>().norm();
}

// -------------------------------------------------------------------------------
// Huber robust-kernel weight (IRLS weight).
//
// Inputs:
//   e : residual norm (>= 0), here in pixels.
//   k : Huber threshold.
// Output:
//   weight w(e) = 1 for e < k (quadratic region), k/e otherwise (linear region).
// The corresponding robust cost used throughout this file is
//   rho(e) = 0.5 * w * (2 - w) * e^2,
// which equals 0.5 e^2 for e < k and k*e - 0.5 k^2 for e >= k.
// -------------------------------------------------------------------------------
inline double huberWeight(double e, double k) { return e < k ? 1.0 : k / e; }

}  // namespace

// ---------------------------------------------------------------------------------
// Constructor.
//
// Inputs:
//   config : estimator tuning parameters (window sizes, LM settings, weights, ...).
//   calib  : camera intrinsics/extrinsics (T_i_c), IMU noise, gravity magnitude.
// Effects:
//   * Pre-computes the bias random-walk weights 1/sigma^2 (per second) for the gyro
//     and accelerometer biases; they are divided by dt per IMU factor in addImu().
//   * Sets the gravity vector in the world frame to (0, 0, -g) (z-up world).
// ---------------------------------------------------------------------------------
VioEstimator::VioEstimator(const VioConfig& config, const Calibration& calib)
    : config_(config), calib_(calib) {
  gyro_bias_weight_ = calib.gyro_bias_std.array().square().inverse();
  accel_bias_weight_ = calib.accel_bias_std.array().square().inverse();
  g_ = Vec3(0, 0, -calib.gravity);
}

// ---------------------------------------------------------------------------------
// Static initialization from a single accelerometer sample.
//
// Inputs:
//   t_ns  : timestamp (ns) of the first state; becomes the first frame id.
//   accel : (bias corrected) accelerometer measurement in the IMU frame while static.
// Effects:
//   Builds a NavState at rest (zero velocity / zero biases / zero position) whose
//   rotation R_w_i aligns the measured specific force with the world +z axis (the
//   accelerometer measures -g, i.e. "up", at rest). Yaw is unobservable and is left
//   arbitrary by FromTwoVectors. Then delegates to initialize(const NavState&).
// ---------------------------------------------------------------------------------
void VioEstimator::initialize(int64_t t_ns, const Vec3& accel) {
  NavState s;
  s.t_ns = t_ns;
  s.T_w_i.R = Quat::FromTwoVectors(accel, Vec3::UnitZ()).toRotationMatrix();
  initialize(s);
}

// ---------------------------------------------------------------------------------
// Initialization from a full state with a user-supplied diagonal prior.
//
// Inputs:
//   state      : complete initial NavState (pose, velocity, biases, timestamp).
//   prior_info : 15 diagonal information values (1/sigma^2) for the increment
//                [dp, dphi(world), dv, dbg, dba] of the first state.
// Effects:
//   Runs the default initialization and then REPLACES Basalt's default prior
//   (position + yaw + biases) by a diagonal prior with the given information,
//   e.g. to hand over a covariance from another estimator.
// ---------------------------------------------------------------------------------
void VioEstimator::initialize(const NavState& state, const Eigen::Matrix<double, 15, 1>& prior_info) {
  initialize(state);
  marg_H_.setZero();
  marg_H_.diagonal() = prior_info;
}

// ---------------------------------------------------------------------------------
// Initialization from a complete state (the common back end of all initialize()).
//
// Input:
//   s : initial NavState; s.t_ns becomes the id of the first window state.
// Effects:
//   * Resets the window so it contains only this one full state.
//   * Creates the initial marginalization prior E(x) = 0.5 d^T H d + b^T d on that
//     state with b = 0 and a diagonal H:
//       - position (indices 0..2) and yaw (index 5 = rotation about world z) get
//         vio_init_pose_weight. These are the 4 directions that are unobservable in
//         VIO (global position + yaw), so the prior fixes the gauge.
//         Roll/pitch (3,4) are observable through gravity and get no prior.
//       - the bias blocks (9..11, 12..14) get small weights.
//         NOTE: following Basalt, index 9..11 (gyro bias in applyInc) is assigned
//         vio_init_ba_weight and 12..14 (accel bias) vio_init_bg_weight; the optional
//         vio_init_gyro/accel_bias_weight overrides use the applyInc ordering.
//   * Stores the linearization point of the prior (marg_lin_) = the initial state.
// ---------------------------------------------------------------------------------
void VioEstimator::initialize(const NavState& s) {
  const int64_t t_ns = s.t_ns;
  frame_states_.clear();
  frame_states_[t_ns] = s;
  last_state_t_ns_ = t_ns;

  // prior covers exactly one full 15-dof state at offset 0
  marg_order_ = AbsOrderMap();
  marg_order_.abs_order_map[t_ns] = {0, POSE_VEL_BIAS_SIZE};
  marg_order_.total_size = POSE_VEL_BIAS_SIZE;
  marg_H_.setZero(POSE_VEL_BIAS_SIZE, POSE_VEL_BIAS_SIZE);
  marg_b_.setZero(POSE_VEL_BIAS_SIZE);
  // prior on position and yaw, small prior on biases (same indices as Basalt)
  marg_H_.diagonal().head<3>().setConstant(config_.vio_init_pose_weight);
  marg_H_(5, 5) = config_.vio_init_pose_weight;
  marg_H_.diagonal().segment<3>(9).setConstant(config_.vio_init_ba_weight);
  marg_H_.diagonal().segment<3>(12).setConstant(config_.vio_init_bg_weight);
  if (config_.vio_init_gyro_bias_weight > 0)
    marg_H_.diagonal().segment<3>(9).setConstant(config_.vio_init_gyro_bias_weight);
  if (config_.vio_init_accel_bias_weight > 0)
    marg_H_.diagonal().segment<3>(12).setConstant(config_.vio_init_accel_bias_weight);
  // the prior is linearized at the initial state (d = 0 right now)
  marg_lin_.clear();
  marg_lin_[t_ns] = s;

  initialized_ = true;
  std::cout << "VIO initialized at t_ns " << t_ns << "\nR_w_i\n" << s.T_w_i.R
            << "\nv_w_i " << s.vel_w_i.transpose() << "\nbg " << s.bias_gyro.transpose()
            << "\nba " << s.bias_accel.transpose() << std::endl;
}

// ---------------------------------------------------------------------------------
// Look up the current estimate of a frame pose.
//
// Input:
//   id : frame timestamp (ns).
// Output:
//   T_w_i (IMU-to-world pose) of that frame, taken from the full states if present,
//   otherwise from the pose-only keyframes. Throws std::out_of_range if the frame is
//   not in the window.
// ---------------------------------------------------------------------------------
SE3 VioEstimator::getPose(int64_t id) const {
  auto it = frame_states_.find(id);
  if (it != frame_states_.end()) return it->second.T_w_i;
  return frame_poses_.at(id).T_w_i;
}

// ---------------------------------------------------------------------------------
// Process one new camera frame (the main entry point, called per stereo frame).
//
// Inputs:
//   of   : optical-flow result of the new frame: timestamp of_->t_ns and, per camera
//          (0 = left, 1 = right), a map KeypointId -> 2-D pixel observation.
//   meas : IMU preintegration from the previous state's time to of->t_ns
//          (may be null, then no new state is predicted).
//   aid  : optional external aiding measurement (baro height, airspeed, course) for
//          this frame (may be null).
// Output:
//   always returns true. Side effects: the window, landmarks, prior and stats_ are
//   updated; lastState() then returns the optimized newest state.
//
// Steps:
//   1. IMU prediction of the new state and storage of the IMU factor.
//   2. Gating of the aiding measurement against the predicted velocity.
//   3. Association of keypoints with existing landmarks (adds observations).
//   4. Keyframe decision and triangulation of new landmarks.
//   5. Collect "lost" landmarks (not tracked in the new frame).
//   6. optimize()  — LM bundle adjustment over the window.
//   7. marginalize() — shrink the window and update the prior.
// ---------------------------------------------------------------------------------
bool VioEstimator::measure(const OpticalFlowResult::Ptr& of, const ImuPreintegration::Ptr& meas,
                           const AidingMeasurement* aid) {
  stats_ = VioStats();
  // --- 1. IMU prediction: new state = previous state propagated by the
  //        preintegrated IMU measurement (gravity g_ in the world frame)
  if (meas) {
    NavState next;
    meas->predictState(frame_states_.at(last_state_t_ns_), g_, next);
    next.t_ns = of->t_ns;
    last_state_t_ns_ = of->t_ns;
    frame_states_[last_state_t_ns_] = next;
    // IMU factor keyed by its start time (connects state startT -> startT + dt)
    imu_meas_[meas->startT()] = *meas;
  }

  // keep the raw flow result: used later to collect all past observations of a
  // keypoint when it is triangulated
  prev_opt_flow_res_[of->t_ns] = of;

  // --- 2. aiding measurement gating
  if (aid) {  // gating against the IMU-predicted state (fixed for the life of the factor)
    AidingMeasurement a = *aid;
    const Vec3& v = frame_states_.at(last_state_t_ns_).vel_w_i;
    const double vh = v.head<2>().norm();
    // speed and course are only meaningful when actually moving (flying)
    const bool in_flight = vh > config_.aid_min_speed && a.airspeed > config_.aid_min_speed;
    a.has_speed = a.has_speed && in_flight;
    if (a.has_course) {
      // innovation between predicted course atan2(vE, vN) and measured course, wrapped to [-pi, pi]
      double innov = std::atan2(v.x(), v.y()) - a.course;
      innov = std::atan2(std::sin(innov), std::cos(innov));
      const bool gate_ok = std::abs(innov) < config_.aid_course_max_innov_deg * M_PI / 180;
      // re-acquisition: after 10 consecutive gate
      // rejections (1 s at 10 Hz) the course is used again (the Huber loss limits its pull)
      if (in_flight && !gate_ok) course_rejected_++;
      if (gate_ok) course_rejected_ = 0;
      a.has_course = in_flight && (gate_ok || course_rejected_ >= 10);
    }
    if (a.has_baro || a.has_speed || a.has_course) aid_[last_state_t_ns_] = a;
    stats_.aid_baro = a.has_baro;
    stats_.aid_speed = a.has_speed;
    stats_.aid_course = a.has_course;
  }

  // --- 3. new residuals for existing landmarks
  //   connected0           : # left-camera keypoints that belong to a known landmark
  //   num_points_connected : per host keyframe, # observations in this frame
  //                          (used by the keyframe marginalization heuristic)
  //   unconnected_obs0     : left-camera keypoints without landmark (triangulation candidates)
  int connected0 = 0;
  std::map<int64_t, int> num_points_connected;
  std::unordered_set<KeypointId> unconnected_obs0;
  for (int i = 0; i < 2; i++) {
    TimeCamId tcid_target(of->t_ns, i);
    for (const auto& kv : of->observations[i]) {
      auto it = landmarks_.find(kv.first);
      if (it != landmarks_.end()) {
        it->second.obs[tcid_target] = kv.second;
        num_points_connected[it->second.host.frame_id]++;
        if (i == 0) connected0++;
      } else if (i == 0) {
        unconnected_obs0.insert(kv.first);
      }
    }
  }

  // --- 4. keyframe decision: too small a fraction of tracked keypoints and enough
  //        frames since the last keyframe -> this frame becomes a keyframe
  if (double(connected0) / double(connected0 + unconnected_obs0.size()) <
          config_.vio_new_kf_keypoints_thresh &&
      frames_after_kf_ > config_.vio_min_frames_after_kf)
    take_kf_ = true;

  if (take_kf_) {
    take_kf_ = false;
    frames_after_kf_ = 0;
    kf_ids_.insert(last_state_t_ns_);
    stats_.is_kf = true;
    // new landmarks are hosted in the LEFT camera of this keyframe
    TimeCamId tcidl(of->t_ns, 0);
    const double min_triang_distance2 =
        config_.vio_min_triangulation_dist * config_.vio_min_triangulation_dist;
    int num_points_added = 0;
    stats_.triang_candidates = static_cast<int>(unconnected_obs0.size());
    for (KeypointId lm_id : unconnected_obs0) {
      // gather every observation of this keypoint in all stored frames / cameras
      // (including the right camera of the current frame -> stereo triangulation)
      std::map<TimeCamId, Vec2> kp_obs;
      for (const auto& kv : prev_opt_flow_res_) {
        for (int k = 0; k < 2; k++) {
          auto it = kv.second->observations[k].find(lm_id);
          if (it != kv.second->observations[k].end()) kp_obs[TimeCamId(kv.first, k)] = it->second;
        }
      }
      bool valid_kp = false;
      bool has_baseline = false;
      const Vec2 p0 = of->observations[0].at(lm_id);
      Vec3 f0;
      if (!calib_.cam[0].unproject(p0, f0)) continue;
      // try observations until one gives a valid triangulation
      for (const auto& kv_obs : kp_obs) {
        if (valid_kp) break;
        const TimeCamId& tcido = kv_obs.first;
        Vec3 f1;
        if (!calib_.cam[tcido.cam_id].unproject(kv_obs.second, f1)) continue;
        // relative pose host-camera <- other-camera:
        //   T_0_1 = (T_i_c0)^-1 * (T_w_i0)^-1 * T_w_i1 * T_i_c1
        const SE3 T_i0_i1 = getPose(tcidl.frame_id).inverse() * getPose(tcido.frame_id);
        const SE3 T_0_1 = calib_.T_i_c[0].inverse() * T_i0_i1 * calib_.T_i_c[tcido.cam_id];
        // require a minimal baseline for a well-conditioned triangulation
        if (T_0_1.t.squaredNorm() < min_triang_distance2) continue;
        has_baseline = true;
        const Vec4 p = triangulate(f0, f1, T_0_1);
        // accept: finite, in front of the camera, and not closer than 1/3 m
        if (p.allFinite() && p[3] > 0 && p[3] < 3.0) {
          Landmark lm;
          lm.host = tcidl;
          lm.dir = Stereographic::project(p.head<3>());
          lm.inv_dist = p[3];
          landmarks_[lm_id] = lm;
          num_points_added++;
          valid_kp = true;
        }
      }
      if (!has_baseline) stats_.triang_no_baseline++;
      // a new landmark receives all of its past observations at once
      if (valid_kp) {
        auto& lm = landmarks_[lm_id];
        for (const auto& kv_obs : kp_obs) lm.obs[kv_obs.first] = kv_obs.second;
      }
    }
    num_points_kf_[of->t_ns] = num_points_added;
    stats_.new_landmarks = num_points_added;
  } else {
    frames_after_kf_++;
  }

  // --- 5. landmarks not observed in either camera of the new frame are "lost"
  //        and (optionally) marginalized out in this step
  std::unordered_set<KeypointId> lost_landmarks;
  if (config_.vio_marg_lost_landmarks) {
    for (const auto& kv : landmarks_) {
      if (!of->observations[0].count(kv.first) && !of->observations[1].count(kv.first))
        lost_landmarks.insert(kv.first);
    }
  }

  // --- 6./7. bundle adjustment and window management
  optimize();
  marginalize(num_points_connected, lost_landmarks);

  stats_.num_landmarks = static_cast<int>(landmarks_.size());
  int nobs = 0;
  for (const auto& kv : landmarks_) nobs += static_cast<int>(kv.second.obs.size());
  stats_.num_obs = nobs;
  stats_.num_kfs = static_cast<int>(kf_ids_.size());
  return true;
}

// ---------------------------------------------------------------------------------
// Reprojection residual and Jacobians of ONE landmark observation.
//
// Inputs:
//   lm     : landmark (host frame/camera, stereographic bearing `dir`, inverse distance rho).
//   tcid_t : target frame/camera in which the landmark is observed.
//   z      : measured pixel coordinates in the target camera.
// Outputs:
//   res : 2-D reprojection residual  r = pi(p_c) - z  (pixels).
//   Jl  : (optional, 2x3) d r / d [d_dir(2), d_rho(1)]  (landmark increment).
//   Jh  : (optional, 2x6) d r / d [dp, dphi] of the HOST frame pose increment.
//   Jt  : (optional, 2x6) d r / d [dp, dphi] of the TARGET frame pose increment.
//   return : false if the point is invalid (rho <= 0 / non-finite), projects outside
//            the valid camera model, or the residual is non-finite.
//
// Geometry (homogeneous point scaled by rho, projection is scale invariant):
//   b   = unproject_stereographic(dir)                    unit bearing, host camera
//   q_h = R_ic_h * b + t_ic_h * rho                       point in host IMU frame (x rho)
//   Pw  = R_wh * q_h + rho * t_wh                         point in world (x rho)
//   D   = Pw - rho * t_wt
//   p_c = R_ic_t^T * (R_wt^T * D - rho * t_ic_t)          point in target camera (x rho)
//   r   = pi(p_c) - z,           Jp = d pi / d p_c (2x3)
//
// Jacobians (left / world-frame perturbation of the poses, see top of file):
//   d p_c / d dp_h   =  A R_wt^T * rho         d p_c / d dphi_h = -A R_wt^T * [R_wh q_h]x
//   d p_c / d dp_t   = -A R_wt^T * rho         d p_c / d dphi_t =  A R_wt^T * [D]x
//   d p_c / d dir    =  A R_wt^T R_wh R_ic_h * (d b / d dir)
//   d p_c / d rho    =  A R_wt^T (R_wh t_ic_h + t_wh - t_wt) - A t_ic_t
// with A = R_ic_t^T. If host and target are the same frame (e.g. the right camera of
// the host keyframe) the relative pose is the identity independent of the frame pose,
// so the pose Jacobians are zero; only the landmark Jacobian is non-zero.
// ---------------------------------------------------------------------------------
bool VioEstimator::linearizePoint(const Landmark& lm, const TimeCamId& tcid_t, const Vec2& z,
                                  Vec2& res, Mat23* Jl, Mat26* Jh, Mat26* Jt) const {
  const double rho = lm.inv_dist;
  if (!(rho > 0) || !std::isfinite(rho)) return false;
  const SE3 Th = getPose(lm.host.frame_id);   // T_w_i of the host frame
  const SE3 Tt = getPose(tcid_t.frame_id);    // T_w_i of the target frame
  const SE3& Tic_h = calib_.T_i_c[lm.host.cam_id];  // host camera extrinsics
  const SE3& Tic_t = calib_.T_i_c[tcid_t.cam_id];   // target camera extrinsics

  // bearing in host camera (+ Jacobian d b / d dir if the landmark Jacobian is needed)
  Eigen::Matrix<double, 3, 2> Jb;
  const Vec3 b = Stereographic::unproject(lm.dir, Jl ? &Jb : nullptr);
  const Vec3 q_h = Tic_h.R * b + Tic_h.t * rho;   // host IMU frame (scaled by rho)
  const Vec3 Rq = Th.R * q_h;                     // rotated into world
  const Vec3 Pw = Rq + rho * Th.t;                // world point (scaled by rho)
  const Vec3 D = Pw - rho * Tt.t;                 // relative to target IMU origin, world axes
  const Mat3 RtT = Tt.R.transpose();
  const Mat3 A = Tic_t.R.transpose();
  const Vec3 pc = A * (RtT * D - rho * Tic_t.t);  // target camera frame (scaled by rho)

  Vec2 uv;
  Mat23 Jp;
  const bool need_j = Jl || Jh || Jt;
  if (!calib_.cam[tcid_t.cam_id].project(pc, uv, need_j ? &Jp : nullptr)) return false;
  res = uv - z;
  if (!res.allFinite()) return false;

  if (need_j) {
    const Mat3 ART = A * RtT;
    const Mat23 JpART = Jp * ART;   // d r / d D
    if (Jl) {
      // columns 0..1: stereographic direction; column 2: inverse distance
      Jl->leftCols<2>() = JpART * Th.R * Tic_h.R * Jb;
      Jl->col(2) = JpART * (Th.R * Tic_h.t + Th.t - Tt.t) - Jp * A * Tic_t.t;
    }
    const bool same_frame = lm.host.frame_id == tcid_t.frame_id;
    if (Jh) {
      if (same_frame) {
        Jh->setZero();
      } else {
        // host translation enters as +rho * t_wh, host rotation as Exp(dphi) * R_wh * q_h
        Jh->leftCols<3>() = JpART * rho;
        Jh->rightCols<3>() = -JpART * hat(Rq);
      }
    }
    if (Jt) {
      if (same_frame) {
        Jt->setZero();
      } else {
        // target translation enters as -rho * t_wt, target rotation as R_wt^T Exp(-dphi) D
        Jt->leftCols<3>() = -JpART * rho;
        Jt->rightCols<3>() = JpART * hat(D);
      }
    }
  }
  return true;
}

// ---------------------------------------------------------------------------------
// Difference between the current estimate and the prior's linearization point.
//
// Inputs (members): marg_order_ (variables in the prior), marg_lin_ (linearization
//   point of each prior variable), frame_states_ / frame_poses_ (current estimate).
// Output:
//   d = x [-] x_lin, stacked in prior order (size marg_order_.total_size):
//     dp   = t - t_lin
//     dphi = Log(R * R_lin^T)          (world-frame rotation difference, consistent
//                                       with the left perturbation in applyInc)
//     dv, dbg, dba = plain differences (only for full 15-dof prior variables)
// ---------------------------------------------------------------------------------
VecX VioEstimator::priorDelta() const {
  VecX d = VecX::Zero(marg_order_.total_size);
  for (const auto& kv : marg_order_.abs_order_map) {
    const int off = kv.second.first, sz = kv.second.second;
    const NavState& lin = marg_lin_.at(kv.first);
    // the variable can now be either a full state or a pose-only keyframe
    const NavState* cur = nullptr;
    auto its = frame_states_.find(kv.first);
    if (its != frame_states_.end())
      cur = &its->second;
    else
      cur = &frame_poses_.at(kv.first);
    d.segment<3>(off) = cur->T_w_i.t - lin.T_w_i.t;
    d.segment<3>(off + 3) = logSO3(cur->T_w_i.R * lin.T_w_i.R.transpose());
    if (sz == POSE_VEL_BIAS_SIZE) {
      d.segment<3>(off + 6) = cur->vel_w_i - lin.vel_w_i;
      d.segment<3>(off + 9) = cur->bias_gyro - lin.bias_gyro;
      d.segment<3>(off + 12) = cur->bias_accel - lin.bias_accel;
    }
  }
  return d;
}

// ---------------------------------------------------------------------------------
// Add the marginalization prior to a linear system.
//
// Inputs:
//   aom : ordering (frame id -> offset, size) of the target system H, b. Every
//         variable of the prior must be present in aom with the same size.
// In/out:
//   H, b  : system to which the prior's Hessian and gradient are added.
//   error : (optional) accumulated cost; the prior's cost is added.
//
// The prior is the quadratic  E(d) = 0.5 d^T H_m d + b_m^T d,  d = x [-] x_lin.
// Its Hessian is H_m (kept FIXED at the linearization point — this is what keeps the
// estimator consistent), and its gradient at the current estimate is
//   dE/dd = b_m + H_m d.
// Blocks are scattered from the prior's own ordering (marg_order_) to aom.
// ---------------------------------------------------------------------------------
void VioEstimator::addPrior(const AbsOrderMap& aom, MatX& H, VecX& b, double* error) const {
  const VecX d = priorDelta();
  const VecX bm = marg_b_ + marg_H_ * d;  // gradient of the prior at the current estimate
  if (error) *error += 0.5 * d.dot(marg_H_ * d) + marg_b_.dot(d);
  for (const auto& kv1 : marg_order_.abs_order_map) {
    const auto& a1 = aom.abs_order_map.at(kv1.first);
    const int o1 = kv1.second.first, s1 = kv1.second.second;
    if (a1.second != s1) {
      std::cerr << "prior/aom size mismatch" << std::endl;
      continue;
    }
    b.segment(a1.first, s1) += bm.segment(o1, s1);
    for (const auto& kv2 : marg_order_.abs_order_map) {
      const auto& a2 = aom.abs_order_map.at(kv2.first);
      const int o2 = kv2.second.first, s2 = kv2.second.second;
      H.block(a1.first, a2.first, s1, s2) += marg_H_.block(o1, o2, s1, s2);
    }
  }
}

// ---------------------------------------------------------------------------------
// Add IMU preintegration factors and bias random-walk factors.
//
// Inputs:
//   aom : ordering of the system; a factor is used only if BOTH of its states are in
//         aom and are full states (in frame_states_).
// In/out:
//   H, b  : (optional, may be null -> error only) system to accumulate into.
//   error : accumulated cost (always updated).
//
// For each IMU factor between states s0 (t0) and s1 (t1 = t0 + dt):
//   * Preintegration residual r (9x1, [r_phi, r_v, r_p]) with information W = info()
//     and Jacobians J0, J1 (9x15) w.r.t. the 15-dof increments of s0 and s1.
//     Contributes the 2x2 block pattern  [J0 J1]^T W [J0 J1]  and  [J0 J1]^T W r.
//   * Bias random walk: r_bg = bg1 - bg0, r_ba = ba1 - ba0 with information
//     1/(sigma^2 * dt) per axis. Jacobians are -I (s0) and +I (s1), which gives the
//     +w/-w diagonal/off-diagonal pattern written explicitly below.
// ---------------------------------------------------------------------------------
void VioEstimator::addImu(const AbsOrderMap& aom, MatX* H, VecX* b, double& error) const {
  for (const auto& kv : imu_meas_) {
    const ImuPreintegration& m = kv.second;
    const int64_t t0 = m.startT(), t1 = m.startT() + m.dtNs();
    auto i0 = aom.abs_order_map.find(t0), i1 = aom.abs_order_map.find(t1);
    if (i0 == aom.abs_order_map.end() || i1 == aom.abs_order_map.end()) continue;
    auto s0 = frame_states_.find(t0), s1 = frame_states_.find(t1);
    if (s0 == frame_states_.end() || s1 == frame_states_.end()) continue;

    Eigen::Matrix<double, 9, 15> J0, J1;
    const auto r = m.residual(s0->second, s1->second, g_, H ? &J0 : nullptr, H ? &J1 : nullptr);
    const Eigen::Matrix<double, 9, 9> W = m.info();
    error += 0.5 * r.dot(W * r);

    // bias random walk residuals and their (per-interval) weights
    const double dt = m.dt();
    const Vec3 res_bg = s1->second.bias_gyro - s0->second.bias_gyro;
    const Vec3 res_ba = s1->second.bias_accel - s0->second.bias_accel;
    const Vec3 wg = gyro_bias_weight_ / dt, wa = accel_bias_weight_ / dt;
    error += 0.5 * res_bg.dot(wg.asDiagonal() * res_bg) + 0.5 * res_ba.dot(wa.asDiagonal() * res_ba);

    if (H) {
      // preintegration factor: H += J^T W J, b += J^T W r for J = [J0 J1]
      const int o0 = i0->second.first, o1 = i1->second.first;
      H->block<15, 15>(o0, o0) += J0.transpose() * W * J0;
      H->block<15, 15>(o0, o1) += J0.transpose() * W * J1;
      H->block<15, 15>(o1, o0) += J1.transpose() * W * J0;
      H->block<15, 15>(o1, o1) += J1.transpose() * W * J1;
      b->segment<15>(o0) += J0.transpose() * W * r;
      b->segment<15>(o1) += J1.transpose() * W * r;
      // bias random walk: r = b1 - b0
      // (indices 9..11 = gyro bias, 12..14 = accel bias in the 15-dof increment)
      for (int k = 0; k < 3; k++) {
        H->coeffRef(o0 + 9 + k, o0 + 9 + k) += wg[k];
        H->coeffRef(o1 + 9 + k, o1 + 9 + k) += wg[k];
        H->coeffRef(o0 + 9 + k, o1 + 9 + k) -= wg[k];
        H->coeffRef(o1 + 9 + k, o0 + 9 + k) -= wg[k];
        (*b)[o0 + 9 + k] -= wg[k] * res_bg[k];
        (*b)[o1 + 9 + k] += wg[k] * res_bg[k];
        H->coeffRef(o0 + 12 + k, o0 + 12 + k) += wa[k];
        H->coeffRef(o1 + 12 + k, o1 + 12 + k) += wa[k];
        H->coeffRef(o0 + 12 + k, o1 + 12 + k) -= wa[k];
        H->coeffRef(o1 + 12 + k, o0 + 12 + k) -= wa[k];
        (*b)[o0 + 12 + k] -= wa[k] * res_ba[k];
        (*b)[o1 + 12 + k] += wa[k] * res_ba[k];
      }
    }
  }
}

// ---------------------------------------------------------------------------------
// Add external aiding factors (barometer, airspeed, course) — all unary on one state.
//
// Inputs:
//   aom      : ordering of the system; only full states present in aom are used.
//   aid_only : (optional) restrict to the aiding factors of these frame ids
//              (used during marginalization); null = all factors.
// In/out:
//   H, b  : (optional, may be null -> error only) system to accumulate into.
//   error : accumulated robust cost.
//
// Every factor is a scalar residual r with std sigma and a 1x15 Jacobian J w.r.t. the
// state increment, robustified with a Huber kernel on the normalized error |r|/sigma
// (threshold config_.aid_huber):
//   baro   : r = p_z - baro_up                              J = d/d p_z = 1
//   speed  : r = |v_h - wind| - airspeed                    J = (v_h - wind)^T / |.|
//   course : r = wrap(atan2(v_E, v_N) - course)             J = [v_N, -v_E] / (v_E^2 + v_N^2)
// Denominators are clamped to >= 1 to avoid huge Jacobians at low speed.
// ---------------------------------------------------------------------------------
void VioEstimator::addAiding(const AbsOrderMap& aom, MatX* H, VecX* b, double& error,
                            const std::set<int64_t>* aid_only) const {
  const double k = config_.aid_huber;
  // add one robustified scalar factor at state offset o
  auto add = [&](int o, const Eigen::Matrix<double, 1, 15>& J, double r, double sigma) {
    const double e = std::abs(r) / sigma;
    const double hw = e < k ? 1.0 : k / e;
    error += 0.5 * hw * (2 - hw) * e * e;
    if (H) {
      const double w = hw / (sigma * sigma);
      H->block<15, 15>(o, o) += J.transpose() * w * J;
      b->segment<15>(o) += J.transpose() * w * r;
    }
  };
  for (const auto& kv : aid_) {
    if (aid_only && !aid_only->count(kv.first)) continue;
    auto ia = aom.abs_order_map.find(kv.first);
    auto is = frame_states_.find(kv.first);
    if (ia == aom.abs_order_map.end() || is == frame_states_.end() || ia->second.second != POSE_VEL_BIAS_SIZE)
      continue;
    const int o = ia->second.first;
    const NavState& s = is->second;
    const AidingMeasurement& a = kv.second;
    Eigen::Matrix<double, 1, 15> J;
    if (a.has_baro) {
      J.setZero();
      J(2) = 1;
      add(o, J, s.T_w_i.t.z() - a.baro_up, config_.aid_baro_sigma);
    }
    if (a.has_speed) {  // |v_h - wind| - TAS
      const Vec2 av(s.vel_w_i.x() - config_.aid_wind_e, s.vel_w_i.y() - config_.aid_wind_n);
      const double n = std::max(av.norm(), 1.0);
      J.setZero();
      J(6) = av.x() / n;
      J(7) = av.y() / n;
      add(o, J, av.norm() - a.airspeed, config_.aid_speed_sigma);
    }
    if (a.has_course) {  // atan2(vE, vN) - course
      const double ve = s.vel_w_i.x(), vn = s.vel_w_i.y(), h2 = std::max(ve * ve + vn * vn, 1.0);
      double r = std::atan2(ve, vn) - a.course;
      r = std::atan2(std::sin(r), std::cos(r));
      J.setZero();
      J(6) = vn / h2;
      J(7) = -ve / h2;
      add(o, J, r, config_.aid_course_sigma_deg * M_PI / 180);
    }
  }
}

// ---------------------------------------------------------------------------------
// LINEARIZATION: build the full Gauss–Newton system at the current estimate.
//
// Inputs:
//   aom      : ordering of the frame variables (frame id -> offset, size 6 or 15).
//   lm_ids   : landmarks whose reprojection factors are included.
//   aid_only : (optional) restriction of the aiding factors (see addAiding).
// Output (ls):
//   ls.H, ls.b : n x n Hessian and n gradient over the FRAME variables only
//                (n = aom.total_size), containing visual + IMU + aiding + prior terms.
//   ls.lms     : per landmark, the landmark-related blocks needed for the Schur
//                complement, kept separately instead of in ls.H:
//                  Hll (3x3) = sum Jl^T w Jl     landmark-landmark block
//                  bl  (3)   = sum Jl^T w r      landmark gradient
//                  Hpl       = list of (frame offset, 6x3 block  sum Jp^T w Jl)
//   ls.error   : total robust cost at the current estimate.
//
// The full system has the arrow-head structure
//     [ Hpp  Hpl ] [dx_p]     [bp]
//     [ Hlp  Hll ] [dx_l] = - [bl]
// with block-diagonal Hll (each landmark is 3x3), so landmarks can be eliminated
// cheaply (see optimize / marginalize).
//
// Visual factors use a Huber kernel on the pixel error (threshold vio_obs_huber_thresh)
// with isotropic pixel noise vio_obs_std_dev: weight w = huber(e)/sigma^2.
// Observations in the host frame itself (e.g. stereo pair of the host keyframe)
// constrain only the landmark (no pose blocks).
// Observations whose frame is not in aom are skipped.
// ---------------------------------------------------------------------------------
void VioEstimator::linearize(const AbsOrderMap& aom, const std::vector<KeypointId>& lm_ids,
                             LinSystem& ls, const std::set<int64_t>* aid_only) const {
  const int n = aom.total_size;
  ls.H.setZero(n, n);
  ls.b.setZero(n);
  ls.error = 0;
  ls.lms.clear();
  ls.lms.reserve(lm_ids.size());

  const double sigma2 = config_.vio_obs_std_dev * config_.vio_obs_std_dev;
  const double k = config_.vio_obs_huber_thresh;

  for (KeypointId id : lm_ids) {
    const Landmark& lm = landmarks_.at(id);
    auto ih = aom.abs_order_map.find(lm.host.frame_id);
    if (ih == aom.abs_order_map.end()) continue;
    LmLin ll;
    ll.id = id;
    int nres = 0;
    for (const auto& ob : lm.obs) {
      auto it = aom.abs_order_map.find(ob.first.frame_id);
      if (it == aom.abs_order_map.end()) continue;
      Vec2 r;
      Mat23 Jl;
      Mat26 Jh, Jt;
      if (!linearizePoint(lm, ob.first, ob.second, r, &Jl, &Jh, &Jt)) continue;
      // robust (IRLS) weight
      const double e = r.norm();
      const double hw = huberWeight(e, k);
      const double w = hw / sigma2;
      ls.error += 0.5 * hw * (2 - hw) * e * e / sigma2;
      nres++;

      // landmark block and gradient
      ll.Hll += Jl.transpose() * w * Jl;
      ll.bl += Jl.transpose() * w * r;
      if (ob.first.frame_id != lm.host.frame_id) {
        // pose-pose blocks between host (oh) and target (ot) frames
        const int oh = ih->second.first, ot = it->second.first;
        ls.H.block<6, 6>(oh, oh) += Jh.transpose() * w * Jh;
        ls.H.block<6, 6>(oh, ot) += Jh.transpose() * w * Jt;
        ls.H.block<6, 6>(ot, oh) += Jt.transpose() * w * Jh;
        ls.H.block<6, 6>(ot, ot) += Jt.transpose() * w * Jt;
        ls.b.segment<6>(oh) += Jh.transpose() * w * r;
        ls.b.segment<6>(ot) += Jt.transpose() * w * r;
        // pose-landmark coupling blocks (first 6 rows of the frame variable = pose)
        ll.block(oh) += Jh.transpose() * w * Jl;
        ll.block(ot) += Jt.transpose() * w * Jl;
      }
    }
    if (nres > 0) ls.lms.push_back(std::move(ll));
  }

  // remaining factors act on frame variables only
  addImu(aom, &ls.H, &ls.b, ls.error);
  addAiding(aom, &ls.H, &ls.b, ls.error, aid_only);
  addPrior(aom, ls.H, ls.b, &ls.error);
}

// ---------------------------------------------------------------------------------
// Robust visual cost of all landmark observations (no linearization).
//
// Inputs:
//   outlier_threshold : pixel error above which an observation is reported.
// Outputs:
//   outliers : (optional) per landmark, list of (observation, error). An error of
//              -2.0 is a sentinel meaning "the landmark itself is bad" — either the
//              observation could not be projected, or the outlier is the HOST
//              observation — and the whole landmark should be removed.
//   return   : sum of Huber costs 0.5 * w (2 - w) e^2 / sigma^2.
// ---------------------------------------------------------------------------------
double VioEstimator::visualError(
    std::map<KeypointId, std::vector<std::pair<TimeCamId, double>>>* outliers,
    double outlier_threshold) const {
  const double sigma2 = config_.vio_obs_std_dev * config_.vio_obs_std_dev;
  const double k = config_.vio_obs_huber_thresh;
  double error = 0;
  for (const auto& kv : landmarks_) {
    const Landmark& lm = kv.second;
    for (const auto& ob : lm.obs) {
      Vec2 r;
      if (!linearizePoint(lm, ob.first, ob.second, r, nullptr, nullptr, nullptr)) {
        if (outliers) (*outliers)[kv.first].emplace_back(ob.first, -2.0);
        continue;
      }
      const double e = r.norm();
      const double hw = huberWeight(e, k);
      error += 0.5 * hw * (2 - hw) * e * e / sigma2;
      if (outliers && e > outlier_threshold)
        (*outliers)[kv.first].emplace_back(ob.first, ob.first != lm.host ? e : -2.0);
    }
  }
  return error;
}

// ---------------------------------------------------------------------------------
// Total cost of the current estimate (visual + IMU + bias RW + aiding + prior).
//
// Output: scalar cost, evaluated with the same ordering as optimize() (pose-only
// keyframes first, then full states). Used by LM to evaluate a candidate step.
// No Jacobians are computed.
// ---------------------------------------------------------------------------------
double VioEstimator::totalError() const {
  AbsOrderMap aom;
  for (const auto& kv : frame_poses_) {
    aom.abs_order_map[kv.first] = {aom.total_size, POSE_SIZE};
    aom.total_size += POSE_SIZE;
  }
  for (const auto& kv : frame_states_) {
    aom.abs_order_map[kv.first] = {aom.total_size, POSE_VEL_BIAS_SIZE};
    aom.total_size += POSE_VEL_BIAS_SIZE;
  }
  double error = visualError(nullptr, 0);
  addImu(aom, nullptr, nullptr, error);
  addAiding(aom, nullptr, nullptr, error, nullptr);
  // prior cost 0.5 d^T H d + b^T d at the current estimate
  const VecX d = priorDelta();
  error += 0.5 * d.dot(marg_H_ * d) + marg_b_.dot(d);
  return error;
}

// ---------------------------------------------------------------------------------
// Save the current estimate (all window states, keyframe poses and landmark
// parameters) so that a rejected LM step can be undone with restore().
// ---------------------------------------------------------------------------------
void VioEstimator::backup() {
  backup_states_ = frame_states_;
  backup_poses_ = frame_poses_;
  for (auto& kv : landmarks_) {
    kv.second.dir_backup = kv.second.dir;
    kv.second.inv_dist_backup = kv.second.inv_dist;
  }
}

// ---------------------------------------------------------------------------------
// Revert the estimate to the last backup() (used when an LM step increases the cost).
// ---------------------------------------------------------------------------------
void VioEstimator::restore() {
  frame_states_ = backup_states_;
  frame_poses_ = backup_poses_;
  for (auto& kv : landmarks_) {
    kv.second.dir = kv.second.dir_backup;
    kv.second.inv_dist = kv.second.inv_dist_backup;
  }
}

// ---------------------------------------------------------------------------------
// Remove a set of observations from a landmark.
//
// Inputs:
//   id  : landmark id (no-op if it does not exist).
//   obs : observations (frame, camera) to remove.
// Effect: the landmark is deleted entirely if fewer than 2 observations remain
// (it can no longer be constrained).
// ---------------------------------------------------------------------------------
void VioEstimator::removeObservations(KeypointId id, const std::set<TimeCamId>& obs) {
  auto it = landmarks_.find(id);
  if (it == landmarks_.end()) return;
  for (const auto& o : obs) it->second.obs.erase(o);
  if (it->second.obs.size() < 2) landmarks_.erase(it);
}

// ---------------------------------------------------------------------------------
// Outlier rejection on the visual observations.
//
// Inputs:
//   threshold   : reprojection error (pixels) above which an observation is an outlier.
//   min_num_obs : minimum number of inlier observations for a landmark to survive.
// Effect:
//   * landmark removed completely if it would keep < min_num_obs inliers, or if any
//     outlier carries the -2.0 sentinel (projection failure / bad host observation);
//   * otherwise only the outlier observations are removed.
// ---------------------------------------------------------------------------------
void VioEstimator::filterOutliers(double threshold, int min_num_obs) {
  std::map<KeypointId, std::vector<std::pair<TimeCamId, double>>> outliers;
  visualError(&outliers, threshold);
  for (const auto& kv : outliers) {
    const int num_obs = static_cast<int>(landmarks_.at(kv.first).obs.size());
    const int num_outliers = static_cast<int>(kv.second.size());
    bool remove = num_obs - num_outliers < min_num_obs;
    for (const auto& o : kv.second)
      if (o.second == -2.0) remove = true;
    if (remove) {
      landmarks_.erase(kv.first);
    } else {
      std::set<TimeCamId> s;
      for (const auto& o : kv.second) s.insert(o.first);
      removeObservations(kv.first, s);
    }
  }
}

// ---------------------------------------------------------------------------------
// Levenberg–Marquardt bundle adjustment over the sliding window.
//
// Inputs (members): current window states, keyframe poses, landmarks, IMU/aiding
//   factors and prior. Optimization starts only once more than 4 full states exist
//   (then stays enabled).
// Outputs: frame_states_, frame_poses_ and landmarks_ are updated in place;
//   stats_.lm_iterations and stats_.final_error are filled.
//
// Variable ordering (aom): pose-only keyframes (6 dof each) first, then full states
// (15 dof each). Landmarks are NOT in aom; they are eliminated per iteration.
//
// One iteration:
//   1. (once, at iteration vio_filter_iteration) remove outliers and re-linearize.
//   2. Linearize (only after an accepted step or outlier filtering).
//   3. Damped Schur complement: for every landmark l with damped
//        Hll~ = Hll + max(lambda * diag(Hll), min_lambda)
//      form the reduced camera system
//        Hs = Hpp - sum_l Hpl Hll~^-1 Hlp,     bs = bp - sum_l Hpl Hll~^-1 bl
//      then damp Hs's diagonal the same way and solve  Hs dx_p = -bs  (LDLT).
//   4. Back-substitute each landmark:  dx_l = -Hll~^-1 (bl + Hlp dx_p).
//   5. Predicted (model) cost decrease from the UNDAMPED quadratic model:
//        l_diff = -(b^T dx + 0.5 dx^T H dx)   over poses and landmarks.
//   6. POSE UPDATE: apply dx_p to all frames (applyPoseInc / applyInc: additive
//      translation, Exp(dphi) * R left-multiplied rotation, additive v / biases) and
//      dx_l additively to the landmarks (dir, inv_dist).
//   7. Gain ratio rel = actual decrease / predicted decrease.
//        accepted (rel > 0): lambda *= max(1/3, 1 - (2 rel - 1)^3)  (Nielsen update),
//                            re-linearize next iteration; stop when the cost
//                            decrease < 1e-6 or max |dx_p| < 1e-4.
//        rejected          : restore() the backup and increase lambda by lambda_vee,
//                            doubling lambda_vee each time; stop at max_lambda.
//      Non-finite solutions are treated like rejected steps.
// ---------------------------------------------------------------------------------
void VioEstimator::optimize() {
  if (!(opt_started_ || frame_states_.size() > 4)) return;
  opt_started_ = true;

  // ordering of the frame variables: keyframe poses (6) then full states (15)
  AbsOrderMap aom;
  for (const auto& kv : frame_poses_) {
    aom.abs_order_map[kv.first] = {aom.total_size, POSE_SIZE};
    aom.total_size += POSE_SIZE;
  }
  for (const auto& kv : frame_states_) {
    aom.abs_order_map[kv.first] = {aom.total_size, POSE_VEL_BIAS_SIZE};
    aom.total_size += POSE_VEL_BIAS_SIZE;
  }
  const int n = aom.total_size;

  // LM damping parameters
  double lambda = config_.vio_lm_lambda_initial;
  const double min_lambda = config_.vio_lm_lambda_min;
  const double max_lambda = config_.vio_lm_lambda_max;
  double lambda_vee = 2.0;

  LinSystem ls;
  bool need_lin = true;
  bool terminated = false;
  bool filtered = false;
  int it = 0;
  double error_total = 0;
  std::vector<KeypointId> lm_ids;

  while (it <= config_.vio_max_iterations && !terminated) {
    // one-time outlier filtering after a few iterations (the estimate is better by then)
    if (it >= config_.vio_filter_iteration && !filtered) {
      filterOutliers(config_.vio_outlier_threshold, 4);
      filtered = true;
      need_lin = true;
    }
    // re-linearize at the current estimate (after an accepted step / filtering)
    if (need_lin) {
      lm_ids.clear();
      for (const auto& kv : landmarks_) lm_ids.push_back(kv.first);
      linearize(aom, lm_ids, ls);
      error_total = ls.error;
      need_lin = false;
    }

    // --- damped Schur complement on landmarks
    MatX Hs = ls.H;
    VecX bs = ls.b;
    std::vector<Mat3> Hll_inv(ls.lms.size());
    for (size_t l = 0; l < ls.lms.size(); l++) {
      const LmLin& ll = ls.lms[l];
      // LM damping of the 3x3 landmark block, then its inverse
      Mat3 Hd = ll.Hll;
      Hd.diagonal() += (ll.Hll.diagonal() * lambda).cwiseMax(min_lambda);
      Hll_inv[l] = Hd.inverse();
      const Mat3& Hi = Hll_inv[l];
      // Hs -= Hpl_i Hll^-1 Hlp_j for every pair of frames observing the landmark,
      // bs -= Hpl_i Hll^-1 bl
      for (const auto& pi : ll.Hpl) {
        const Mat63 HiHi = pi.second * Hi;
        bs.segment<6>(pi.first) -= HiHi * ll.bl;
        for (const auto& pj : ll.Hpl)
          Hs.block<6, 6>(pi.first, pj.first) -= HiHi * pj.second.transpose();
      }
    }
    // LM damping of the reduced camera system and solve Hs dx_p = -bs
    Hs.diagonal() += (Hs.diagonal() * lambda).cwiseMax(min_lambda);
    VecX inc = -Hs.ldlt().solve(bs);
    if (!inc.allFinite()) {
      // numerical failure: increase damping and retry
      lambda = lambda_vee * lambda;
      lambda_vee *= 2;
      it++;
      if (lambda > max_lambda) terminated = true;
      continue;
    }

    // back-substitute landmarks + compute model cost change (undamped model)
    //   model = b^T dx + 0.5 dx^T H dx, split into pose part and per-landmark part
    //   (bl^T dl + dl^T Hlp dp + 0.5 dl^T Hll dl)
    std::vector<Vec3> inc_l(ls.lms.size());
    double model = ls.b.dot(inc) + 0.5 * inc.dot(ls.H * inc);
    for (size_t l = 0; l < ls.lms.size(); l++) {
      const LmLin& ll = ls.lms[l];
      Vec3 rhs = ll.bl;
      Vec3 Hlp_dp = Vec3::Zero();
      for (const auto& p : ll.Hpl) Hlp_dp += p.second.transpose() * inc.segment<6>(p.first);
      rhs += Hlp_dp;
      // dl = -Hll~^-1 (bl + Hlp dp)
      inc_l[l] = -Hll_inv[l] * rhs;
      model += ll.bl.dot(inc_l[l]) + inc_l[l].dot(Hlp_dp) + 0.5 * inc_l[l].dot(ll.Hll * inc_l[l]);
    }
    const double l_diff = -model;  // predicted cost decrease

    // --- pose / state / landmark update (keep a backup for rejection)
    backup();
    // keyframes: 6-dof [dp, dphi], states: 15-dof [dp, dphi, dv, dbg, dba]
    for (auto& kv : frame_poses_)
      kv.second.applyPoseInc(inc.segment<6>(aom.abs_order_map.at(kv.first).first));
    for (auto& kv : frame_states_)
      kv.second.applyInc(inc.segment<15>(aom.abs_order_map.at(kv.first).first));
    // landmarks: additive update of stereographic direction and inverse distance
    for (size_t l = 0; l < ls.lms.size(); l++) {
      Landmark& lm = landmarks_.at(ls.lms[l].id);
      lm.dir += inc_l[l].head<2>();
      lm.inv_dist += inc_l[l][2];
    }
    double step_norminf = inc.size() ? inc.cwiseAbs().maxCoeff() : 0.0;

    // --- accept / reject by comparing actual and predicted cost decrease
    const double after_error = totalError();
    const double f_diff = error_total - after_error;
    const double rel = f_diff / l_diff;
    if (l_diff > 0 && rel > 0 && std::isfinite(after_error)) {
      // accepted: decrease damping (Nielsen), re-linearize at the new estimate
      lambda *= std::max(1.0 / 3, 1 - std::pow(2 * rel - 1, 3));
      lambda = std::max(min_lambda, lambda);
      lambda_vee = 2.0;
      it++;
      need_lin = true;
      // convergence: negligible cost decrease or tiny step
      if ((f_diff > 0 && f_diff < 1e-6) || step_norminf < 1e-4) terminated = true;
    } else {
      // rejected: undo the step and increase damping
      lambda = lambda_vee * lambda;
      lambda_vee *= 2;
      restore();
      it++;
      if (lambda > max_lambda) terminated = true;
    }
  }
  (void)n;
  stats_.lm_iterations = it;
  stats_.final_error = totalError();
}

// ---------------------------------------------------------------------------------
// MARGINALIZATION: shrink the sliding window and update the Gaussian prior.
//
// Inputs:
//   num_points_connected : per keyframe id, # of its landmarks observed in the newest
//                          frame (from measure()); used to pick keyframes to drop.
//   lost_landmarks       : landmarks not observed in the newest frame; they are
//                          marginalized out (if vio_marg_lost_landmarks).
// Effects:
//   frame_states_, frame_poses_, kf_ids_, landmarks_, imu_meas_, aid_ and the prior
//   (marg_H_, marg_b_, marg_order_, marg_lin_) are updated.
//
// Runs only if there are more than vio_max_kfs keyframe poses or at least
// vio_max_states full states. Then:
//
//   1. States to remove: the oldest full states so that max_states - 1 remain.
//      `last_state_to_marg` is the newest of the old states; it stays a full state and
//      becomes the anchor of the new prior. Every older state is either
//        - a keyframe      -> its velocity/biases are marginalized, its pose kept as a
//                             pose-only keyframe (states_to_marg_vel_bias), or
//        - a non-keyframe  -> marginalized completely (states_to_marg_all).
//      Pose-only frames that are no longer keyframes are marginalized (poses_to_marg).
//   2. Keyframe selection (Basalt/DSO heuristic) while there are too many keyframes
//      (the two newest keyframes are never chosen):
//        a) first keyframe whose ratio (#points seen in newest frame / #points created)
//           is below vio_kf_marg_feature_ratio, else
//        b) the keyframe minimizing the DSO distance score
//             sqrt(d(kf, last_kf)) * sum_j 1 / (d(kf, kf_j) + eps)
//           (drops keyframes that are close to others but far from the newest one).
//   3. Linearize, over the variables in aom (keyframes + states up to
//      last_state_to_marg), the prior, the IMU factors, the aiding factors of the
//      removed states, and the reprojection factors of landmarks hosted in removed
//      keyframes and of lost landmarks.
//   4. Eliminate those landmarks by an (undamped) Schur complement.
//   5. Split the frame variables into keep (k) and marginalize (m) and compute the
//      Schur complement
//        H_new = Hkk - Hkm Hmm^-1 Hmk,       b_new = bk - Hkm Hmm^-1 bm
//      using a pseudo-inverse of Hmm (eigen-decomposition, null directions dropped).
//   6. Remove the marginalized variables / factors / landmarks / observations.
//   7. The new prior is over all remaining keyframe poses (6 dof) + last_state_to_marg
//      (15 dof), with linearization point = current estimate (so d = 0 now and b_new
//      is the gradient at that point).
// ---------------------------------------------------------------------------------
void VioEstimator::marginalize(const std::map<int64_t, int>& num_points_connected,
                               const std::unordered_set<KeypointId>& lost_landmarks) {
  if (!opt_started_) return;
  const int max_states = config_.vio_max_states;
  const size_t max_kfs = static_cast<size_t>(config_.vio_max_kfs);
  if (!(frame_poses_.size() > max_kfs || static_cast<int>(frame_states_.size()) >= max_states))
    return;

  // --- 1. which full states leave the window
  const int states_to_remove = static_cast<int>(frame_states_.size()) - max_states + 1;
  auto sit = frame_states_.cbegin();
  for (int i = 0; i < states_to_remove; i++) sit++;
  const int64_t last_state_to_marg = sit->first;

  // ordering for the marginalization system: all keyframe poses, then the full
  // states up to and including last_state_to_marg (newer states are not involved)
  AbsOrderMap aom;
  std::set<int64_t> poses_to_marg;
  for (const auto& kv : frame_poses_) {
    aom.abs_order_map[kv.first] = {aom.total_size, POSE_SIZE};
    if (kf_ids_.count(kv.first) == 0) poses_to_marg.insert(kv.first);
    aom.total_size += POSE_SIZE;
  }
  std::set<int64_t> states_to_marg_vel_bias, states_to_marg_all;
  for (const auto& kv : frame_states_) {
    if (kv.first > last_state_to_marg) break;
    if (kv.first != last_state_to_marg) {
      if (kf_ids_.count(kv.first) > 0)
        states_to_marg_vel_bias.insert(kv.first);   // keyframe: keep pose, drop v/bg/ba
      else
        states_to_marg_all.insert(kv.first);        // non-keyframe: drop everything
    }
    aom.abs_order_map[kv.first] = {aom.total_size, POSE_VEL_BIAS_SIZE};
    aom.total_size += POSE_VEL_BIAS_SIZE;
  }

  // --- 2. keyframe selection (Basalt / DSO heuristic)
  std::set<int64_t> kfs_to_marg;
  while (kf_ids_.size() > max_kfs && !states_to_marg_vel_bias.empty()) {
    int64_t id_to_marg = -1;
    // a) keyframe with too few of its points still visible in the newest frame
    if (kf_ids_.size() > 2) {
      auto end_minus_2 = std::prev(kf_ids_.end(), 2);
      for (auto it = kf_ids_.begin(); it != end_minus_2; ++it) {
        auto c = num_points_connected.find(*it);
        auto nk = num_points_kf_.find(*it);
        const double npk = nk != num_points_kf_.end() ? std::max(1, nk->second) : 1;
        if (c == num_points_connected.end() || c->second / npk < config_.vio_kf_marg_feature_ratio) {
          id_to_marg = *it;
          break;
        }
      }
    }
    // b) DSO distance score: prefer keyframes close to the others, far from the newest
    if (kf_ids_.size() > 2 && id_to_marg < 0) {
      auto end_minus_2 = std::prev(kf_ids_.end(), 2);
      const int64_t last_kf = *kf_ids_.crbegin();
      double min_score = std::numeric_limits<double>::max();
      int64_t min_id = -1;
      for (auto it1 = kf_ids_.begin(); it1 != end_minus_2; ++it1) {
        double denom = 0;
        for (auto it2 = kf_ids_.begin(); it2 != end_minus_2; ++it2)
          denom += 1 / ((getPose(*it1).t - getPose(*it2).t).norm() + 1e-5);
        const double score = std::sqrt((getPose(*it1).t - getPose(last_kf).t).norm()) * denom;
        if (score < min_score) {
          min_id = *it1;
          min_score = score;
        }
      }
      id_to_marg = min_id;
    }
    if (id_to_marg < 0) break;
    kfs_to_marg.insert(id_to_marg);
    poses_to_marg.insert(id_to_marg);
    kf_ids_.erase(id_to_marg);
  }

  // a selected keyframe that is still a full state is marginalized completely
  for (int64_t id : kfs_to_marg) {
    if (states_to_marg_vel_bias.erase(id)) states_to_marg_all.insert(id);
  }

  // --- 3. linearize: prior + IMU + landmarks hosted in marginalized KFs + lost landmarks
  std::vector<KeypointId> lm_ids;
  for (const auto& kv : landmarks_) {
    if (kfs_to_marg.count(kv.second.host.frame_id) || lost_landmarks.count(kv.first))
      lm_ids.push_back(kv.first);
  }
  // aiding factors: only those of the states that leave the window as full states (their information
  // goes into the prior); the factors of last_state_to_marg stay in the optimization
  std::set<int64_t> aid_marg = states_to_marg_all;
  aid_marg.insert(states_to_marg_vel_bias.begin(), states_to_marg_vel_bias.end());
  LinSystem ls;
  linearize(aom, lm_ids, ls, &aid_marg);

  // --- 4. eliminate the involved landmarks (undamped Schur complement):
  //        H -= Hpl Hll^-1 Hlp,  b -= Hpl Hll^-1 bl
  //        landmarks with a (near) singular Hll are skipped (their info is dropped)
  MatX H = ls.H;
  VecX b = ls.b;
  for (const LmLin& ll : ls.lms) {
    Mat3 Hll = ll.Hll;
    Hll.diagonal().array() += 1e-9;
    Eigen::LDLT<Mat3> ldlt(Hll);
    if (ldlt.info() != Eigen::Success || !(ldlt.vectorD().array() > 1e-12).all()) continue;
    const Mat3 Hi = ldlt.solve(Mat3::Identity());
    if (!Hi.allFinite()) continue;
    for (const auto& pi : ll.Hpl) {
      const Mat63 HiHi = pi.second * Hi;
      b.segment<6>(pi.first) -= HiHi * ll.bl;
      for (const auto& pj : ll.Hpl) H.block<6, 6>(pi.first, pj.first) -= HiHi * pj.second.transpose();
    }
  }

  // --- 5. indices to keep / marginalize
  //   pose-only frame       : 6 indices, keep or marg
  //   state, marg all       : 15 indices marg
  //   state, marg vel/bias  : pose (0..5) kept, v/bg/ba (6..14) marg
  //   other states          : 15 indices kept (only last_state_to_marg here)
  std::vector<int> idx_keep, idx_marg;
  for (const auto& kv : aom.abs_order_map) {
    const int start = kv.second.first;
    if (kv.second.second == POSE_SIZE) {
      for (int i = 0; i < POSE_SIZE; i++)
        (poses_to_marg.count(kv.first) ? idx_marg : idx_keep).push_back(start + i);
    } else if (states_to_marg_all.count(kv.first)) {
      for (int i = 0; i < POSE_VEL_BIAS_SIZE; i++) idx_marg.push_back(start + i);
    } else if (states_to_marg_vel_bias.count(kv.first)) {
      for (int i = 0; i < POSE_SIZE; i++) idx_keep.push_back(start + i);
      for (int i = POSE_SIZE; i < POSE_VEL_BIAS_SIZE; i++) idx_marg.push_back(start + i);
    } else {
      for (int i = 0; i < POSE_VEL_BIAS_SIZE; i++) idx_keep.push_back(start + i);
    }
  }
  // kept indices sorted by timestamp order -> matches the new prior's ordering
  // (keyframe poses by time, then last_state_to_marg) built at the end
  std::sort(idx_keep.begin(), idx_keep.end());
  const int nk = static_cast<int>(idx_keep.size()), nm = static_cast<int>(idx_marg.size());
  // gather the keep/marg sub-blocks
  MatX Hkk(nk, nk), Hkm(nk, nm), Hmm(nm, nm);
  VecX bk(nk), bmv(nm);
  for (int i = 0; i < nk; i++) {
    bk[i] = b[idx_keep[i]];
    for (int j = 0; j < nk; j++) Hkk(i, j) = H(idx_keep[i], idx_keep[j]);
    for (int j = 0; j < nm; j++) Hkm(i, j) = H(idx_keep[i], idx_marg[j]);
  }
  for (int i = 0; i < nm; i++) {
    bmv[i] = b[idx_marg[i]];
    for (int j = 0; j < nm; j++) Hmm(i, j) = H(idx_marg[i], idx_marg[j]);
  }
  // Schur complement: H_new = Hkk - Hkm Hmm^+ Hmk,  b_new = bk - Hkm Hmm^+ bm
  MatX H_new = Hkk;
  VecX b_new = bk;
  if (nm > 0) {
    // robust inverse of Hmm (eigen-decomposition, drop null directions)
    Eigen::SelfAdjointEigenSolver<MatX> es(0.5 * (Hmm + Hmm.transpose()));
    VecX ev = es.eigenvalues();
    const double emax = std::max(ev.maxCoeff(), 1e-12);
    VecX evinv(nm);
    for (int i = 0; i < nm; i++) evinv[i] = ev[i] > emax * 1e-12 ? 1.0 / ev[i] : 0.0;
    const MatX Hmm_inv = es.eigenvectors() * evinv.asDiagonal() * es.eigenvectors().transpose();
    H_new -= Hkm * Hmm_inv * Hkm.transpose();
    b_new -= Hkm * Hmm_inv * bmv;
  }
  // enforce exact symmetry against round-off
  H_new = 0.5 * (H_new + H_new.transpose());

  // --- 6. update window
  for (int64_t id : aid_marg) aid_.erase(id);
  // fully marginalized states: drop the state, its outgoing IMU factor, its flow data
  for (int64_t id : states_to_marg_all) {
    frame_states_.erase(id);
    imu_meas_.erase(id);
    prev_opt_flow_res_.erase(id);
  }
  // keyframe states: demote to pose-only keyframes (velocity/biases are gone)
  for (int64_t id : states_to_marg_vel_bias) {
    frame_poses_[id] = frame_states_.at(id);
    frame_states_.erase(id);
    imu_meas_.erase(id);
  }
  // marginalized keyframe poses
  for (int64_t id : poses_to_marg) {
    frame_poses_.erase(id);
    prev_opt_flow_res_.erase(id);
    num_points_kf_.erase(id);
  }
  // landmark database: remove landmarks hosted in marginalized KFs and
  // observations in removed frames
  for (auto it = landmarks_.begin(); it != landmarks_.end();) {
    if (kfs_to_marg.count(it->second.host.frame_id)) {
      it = landmarks_.erase(it);
      continue;
    }
    for (auto o = it->second.obs.begin(); o != it->second.obs.end();) {
      const int64_t f = o->first.frame_id;
      if (poses_to_marg.count(f) || states_to_marg_all.count(f) || kfs_to_marg.count(f))
        o = it->second.obs.erase(o);
      else
        ++o;
    }
    if (it->second.obs.size() < 2)
      it = landmarks_.erase(it);
    else
      ++it;
  }
  if (config_.vio_marg_lost_landmarks)
    for (KeypointId id : lost_landmarks) landmarks_.erase(id);

  // --- 7. new prior: all keyframe poses + last_state_to_marg, linearized at the
  // current estimate (b_new is the gradient at the current estimate)
  AbsOrderMap order_new;
  marg_lin_.clear();
  for (const auto& kv : frame_poses_) {
    order_new.abs_order_map[kv.first] = {order_new.total_size, POSE_SIZE};
    order_new.total_size += POSE_SIZE;
    marg_lin_[kv.first] = kv.second;
  }
  order_new.abs_order_map[last_state_to_marg] = {order_new.total_size, POSE_VEL_BIAS_SIZE};
  order_new.total_size += POSE_VEL_BIAS_SIZE;
  marg_lin_[last_state_to_marg] = frame_states_.at(last_state_to_marg);

  // sanity check: kept dimension must match the new prior's ordering
  if (order_new.total_size != nk) {
    std::cerr << "Marginalization order mismatch " << order_new.total_size << " vs " << nk
              << std::endl;
  }
  marg_H_ = H_new;
  marg_b_ = b_new;
  marg_order_ = order_new;
}

}  // namespace bw

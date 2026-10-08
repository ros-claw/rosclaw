// Protected generated-model API carrier and observer. These methods forward to
// installed MoveIt APIs; no controller, path-search or grasp algorithm is authored here.
#pragma once
#include <fstream>
#include <sstream>
#include <nlohmann/json.hpp>
#include <urdf_parser/urdf_parser.h>
#include <srdfdom/model.h>
#include <rclcpp/rclcpp.hpp>
#include <pluginlib/class_loader.hpp>
#include <moveit/robot_model/robot_model.hpp>
#include <moveit/robot_state/robot_state.hpp>
#include <moveit/robot_state/conversions.hpp>
#include <moveit/robot_state/cartesian_interpolator.hpp>
#include <moveit/planning_scene/planning_scene.hpp>
#include <moveit/planning_interface/planning_interface.hpp>
#include <moveit/kinematic_constraints/utils.hpp>
using json=nlohmann::json;
inline std::string fixture_read(const char*p){std::ifstream f(p);if(!f)throw std::runtime_error("INPUT_OPEN");std::stringstream b;b<<f.rdbuf();return b.str();}
class MoveItFixture{
  moveit::core::RobotModelPtr model;
  rclcpp::Node::SharedPtr node;
  std::shared_ptr<pluginlib::ClassLoader<kinematics::KinematicsBase>> kl;
  std::shared_ptr<pluginlib::ClassLoader<planning_interface::PlannerManager>> pl;
  kinematics::KinematicsBasePtr kdl;
  planning_interface::PlannerManagerPtr planner;
  planning_scene::PlanningScenePtr scene;
  json records=json::array();
  unsigned planning_calls=0,cartesian_calls=0,state_calls=0;
  void record(const std::string&api,const json&input,const json&output){records.push_back({{"api",api},{"input",input},{"output",output}});}
  moveit::core::RobotState state(const std::vector<double>&q){
    if(q.size()!=6||++state_calls>10000)throw std::runtime_error("FINITE_STATE_BOUND");
    for(double v:q)if(!std::isfinite(v))throw std::runtime_error("FINITE_STATE_VALUE");
    moveit::core::RobotState s(model);s.setToDefaultValues();s.setJointGroupPositions("arm",q);s.update();return s;
  }
  json state_row(const moveit::core::RobotState&s){
    // Library-produced waypoints may have dirty link caches. Refresh a copy:
    // preserve all queried joint values and leave the caller's state untouched.
    moveit::core::RobotState fresh(s);
    fresh.updateLinkTransforms();
    const moveit::core::RobotState& observed=fresh;
    std::vector<double>q;observed.copyJointGroupPositions("arm",q);auto t=observed.getGlobalLinkTransform("tool");json R=json::array();for(int i=0;i<3;i++)R.push_back({t(i,0),t(i,1),t(i,2)});return {{"q",q},{"tool_position_m",{t(0,3),t(1,3),t(2,3)}},{"tool_rotation",R},{"bounds",observed.satisfiesBounds(model->getJointModelGroup("arm"))}};
  }
public:
  MoveItFixture(const char*u,const char*r){
    auto urdf=urdf::parseURDF(fixture_read(u));if(!urdf)throw std::runtime_error("URDF_PARSE");auto srdf=std::make_shared<srdf::Model>();if(!srdf->initString(*urdf,fixture_read(r)))throw std::runtime_error("SRDF_PARSE");model=std::make_shared<moveit::core::RobotModel>(urdf,srdf);
    auto opts=rclcpp::NodeOptions().start_parameter_services(false).start_parameter_event_publisher(false);node=std::make_shared<rclcpp::Node>("finite_moveit_generated_cases",opts);
    kl=std::make_shared<pluginlib::ClassLoader<kinematics::KinematicsBase>>("moveit_core","kinematics::KinematicsBase");kdl=kl->createSharedInstance("kdl_kinematics_plugin/KDLKinematicsPlugin");if(!kdl->initialize(node,*model,"arm","base_link",{"tool"},.01))throw std::runtime_error("KDL_INITIALIZE");
    model->setKinematicsAllocators({{"arm",[this](const moveit::core::JointModelGroup*){return kdl;}}});
    pl=std::make_shared<pluginlib::ClassLoader<planning_interface::PlannerManager>>("moveit_core","planning_interface::PlannerManager");planner=pl->createSharedInstance("ompl_interface/OMPLPlanner");if(!planner->initialize(model,node,"bounded_ompl"))throw std::runtime_error("OMPL_INITIALIZE");scene=std::make_shared<planning_scene::PlanningScene>(model);
  }
  void begin_case(){records=json::array();}
  json trace()const{return records;}
  json metadata(){json joints=json::array();for(auto*j:model->getJointModels())joints.push_back({{"name",j->getName()},{"type",j->getType()},{"variables",j->getVariableNames()},{"first_variable_index",model->getVariableIndex(j->getVariableNames().empty()?model->getVariableNames()[0]:j->getVariableNames()[0])}});json x={{"variables",model->getVariableNames()},{"group_variables",model->getJointModelGroup("arm")->getVariableNames()},{"joints",joints},{"KDL_joint_names",kdl->getJointNames()},{"KDL_loaded",true},{"OMPL_loaded",true}};record("RobotModel.metadata",json::object(),x);return x;}
  json transform(const std::vector<double>&q,const std::string&link="tool"){
    if(!model->hasLinkModel(link)){json x={{"error","UNKNOWN_FRAME"}};record("RobotModel.hasLinkModel",{{"link",link}},x);return x;}
    auto s=state(q);auto t=s.getGlobalLinkTransform(link);json R=json::array();for(int i=0;i<3;i++)R.push_back({t(i,0),t(i,1),t(i,2)});json x={{"position_m",{t(0,3),t(1,3),t(2,3)}},{"rotation",R},{"state",state_row(s)}};record("RobotState.getGlobalLinkTransform",{{"q",q},{"link",link}},x);return x;
  }
  json sphere(const std::string&id,const std::vector<double>&xyz,double radius){
    if(id!="generated_obstacle"||xyz.size()!=3||!(radius>0&&radius<=.2))throw std::runtime_error("GENERATED_SCENE_INPUT");moveit_msgs::msg::CollisionObject o;o.id=id;o.header.frame_id=model->getModelFrame();o.operation=o.ADD;o.pose.orientation.w=1;shape_msgs::msg::SolidPrimitive shape;shape.type=shape.SPHERE;shape.dimensions={radius};geometry_msgs::msg::Pose p;p.orientation.w=1;p.position.x=xyz[0];p.position.y=xyz[1];p.position.z=xyz[2];o.primitives.push_back(shape);o.primitive_poses.push_back(p);if(!scene->processCollisionObjectMsg(o))throw std::runtime_error("SCENE_OBJECT");moveit_msgs::msg::PlanningScene saved;scene->getPlanningSceneMsg(saved);auto obj=scene->getWorld()->getObject(id);if(!obj||obj->shapes_.size()!=1||obj->global_shape_poses_.size()!=1)throw std::runtime_error("ACTUAL_WORLD_OBJECT_SHAPE");auto actual=std::dynamic_pointer_cast<const shapes::Sphere>(obj->shapes_[0]);if(!actual)throw std::runtime_error("ACTUAL_WORLD_NOT_SPHERE");auto pos=obj->global_shape_poses_[0].translation();json x={{"id",obj->id_},{"center_m",{pos[0],pos[1],pos[2]}},{"radius_m",actual->radius},{"world_contains",true}};record("PlanningScene.processCollisionObjectMsg",{{"id",id},{"center_m",xyz},{"radius_m",radius}},x);return x;
  }
  void clear_world(){scene->getWorldNonConst()->clearObjects();record("World.clearObjects",json::object(),json::object());}
  json collision(const std::vector<double>&q){auto s=state(q);collision_detection::CollisionRequest req;req.group_name="arm";collision_detection::CollisionResult res;scene->checkCollision(req,res,s);json x={{"collision",res.collision},{"state",state_row(s)}};record("PlanningScene.checkCollision",{{"q",q}},x);return x;}
  json plan(const std::vector<double>&start,const std::vector<double>&goal,double seconds=.5){
    if(++planning_calls>6||seconds<=0||seconds>.5)throw std::runtime_error("FINITE_PLANNER_BOUND");auto a=state(start),b=state(goal);scene->setCurrentState(a);planning_interface::MotionPlanRequest req;req.group_name="arm";req.allowed_planning_time=seconds;req.num_planning_attempts=1;moveit::core::robotStateToRobotStateMsg(a,req.start_state);req.goal_constraints.push_back(kinematic_constraints::constructGoalConstraints(b,model->getJointModelGroup("arm"),1e-6));moveit_msgs::msg::MoveItErrorCodes ec;auto ctx=planner->getPlanningContext(scene,req,ec);planning_interface::MotionPlanResponse result;if(ctx)ctx->solve(result);bool ok=ctx&&bool(result.error_code);json points=json::array();if(ok&&result.trajectory){if(result.trajectory->getWayPointCount()>1000)throw std::runtime_error("FINITE_PATH_BOUND");for(std::size_t i=0;i<result.trajectory->getWayPointCount();i++)points.push_back(state_row(result.trajectory->getWayPoint(i)));}json x={{"solved",ok},{"error_code",ctx?result.error_code.val:ec.val},{"points",points}};record("OMPLPlanner.getPlanningContext.solve",{{"start",start},{"goal",goal},{"planning_time_s",seconds}},x);return x;
  }
  json cartesian(const std::vector<double>&start,const std::vector<double>&target,double step){
    if(++cartesian_calls>2||target.size()!=3||step<.01||step>.05)throw std::runtime_error("FINITE_CARTESIAN_BOUND");auto s=state(start);Eigen::Isometry3d t=s.getGlobalLinkTransform("tool");t.translation()=Eigen::Vector3d(target[0],target[1],target[2]);std::vector<moveit::core::RobotStatePtr> path;moveit::core::GroupStateValidityCallbackFn valid=[](moveit::core::RobotState*st,const moveit::core::JointModelGroup*g,const double*q){st->setJointGroupPositions(g,q);return st->satisfiesBounds(g);};auto f=moveit::core::CartesianInterpolator::computeCartesianPath(&s,model->getJointModelGroup("arm"),path,model->getLinkModel("tool"),t,true,moveit::core::MaxEEFStep(step),moveit::core::CartesianPrecision(),valid,kinematics::KinematicsQueryOptions());if(path.size()>1000)throw std::runtime_error("FINITE_CARTESIAN_PATH_BOUND");json points=json::array();for(auto&p:path)points.push_back(state_row(*p));json x={{"fraction",double(f)},{"points",points}};record("CartesianInterpolator.computeCartesianPath.KDL",{{"start",start},{"target_m",target},{"step_m",step}},x);return x;
  }
};

// Finite software assertions for the protected generated MoveIt model.
// All measurements originate in the supplied public interface; no trace is authored here.
#include <algorithm>
#include <array>
#include <cmath>
#include <set>
#include <string>
#include <vector>
#include <stdexcept>

namespace finite_moveit {
using V = std::vector<double>;
constexpr double eps = 1e-6;
struct Checks {
  json rows = json::array();
  bool passed = true;
  void test(const std::string& name, bool ok) {
    rows.push_back({{"criterion", name}, {"passed", ok}});
    passed = passed && ok;
  }
  json result(json measured = json::object()) const {
    measured["passed"] = passed;
    measured["criteria"] = rows;
    return measured;
  }
};
bool finite(const V& a, size_t n) {
  return a.size() == n && std::all_of(a.begin(), a.end(), [](double x){return std::isfinite(x);});
}
double distance(const V& a, const V& b) {
  if (a.size() != b.size()) return INFINITY;
  double d=0; for(size_t i=0;i<a.size();++i) d += (a[i]-b[i])*(a[i]-b[i]);
  return std::sqrt(d);
}
bool near(const V& a, const V& b, double tol=eps) {
  return finite(a,b.size()) && finite(b,b.size()) && distance(a,b)<=tol;
}
bool bounds(const V& q) {
  const V lo={-1,-1,.1,-3.14,-1.4,-3.14}, hi={1,1,1,3.14,1.4,3.14};
  if(!finite(q,6)) return false;
  for(size_t i=0;i<6;++i) if(q[i]<lo[i]-eps || q[i]>hi[i]+eps) return false;
  return true;
}
Eigen::Matrix3d rotation(const json& R) {
  if(!R.is_array() || R.size()!=3) throw std::runtime_error("ROTATION_DIMENSION");
  Eigen::Matrix3d m;
  for(int i=0;i<3;++i){V row=R.at(i).get<V>();if(!finite(row,3)) throw std::runtime_error("ROTATION_NONFINITE");for(int j=0;j<3;++j)m(i,j)=row[j];}
  return m;
}
Eigen::Matrix3d fk_rotation(const V& q) {
  return (Eigen::AngleAxisd(q.at(3),Eigen::Vector3d::UnitX()) *
          Eigen::AngleAxisd(q.at(4),Eigen::Vector3d::UnitY()) *
          Eigen::AngleAxisd(q.at(5),Eigen::Vector3d::UnitZ())).toRotationMatrix();
}
bool proper(const Eigen::Matrix3d& r) {
  return r.allFinite() && (r.transpose()*r-Eigen::Matrix3d::Identity()).norm()<eps && std::abs(r.determinant()-1)<eps;
}
bool valid_row(const json& row) {
  V q=row.at("q").get<V>(), p=row.at("tool_position_m").get<V>();
  if(!bounds(q) || !row.at("bounds").get<bool>()) return false;
  auto r=rotation(row.at("tool_rotation"));
  return near(p,V(q.begin(),q.begin()+3),1e-5) && proper(r) && (r-fk_rotation(q)).norm()<1e-5;
}
double segment_distance(const V& a,const V& b,const V& c) {
  double vv=0, dot=0; for(size_t i=0;i<3;++i){double v=b[i]-a[i];vv+=v*v;dot+=(c[i]-a[i])*v;}
  double u=vv>0?std::max(0.,std::min(1.,dot/vv)):0.;
  V closest(3);for(size_t i=0;i<3;++i)closest[i]=a[i]+u*(b[i]-a[i]);
  return distance(closest,c);
}
bool object_matches(const json& o,const V& center,double radius,const std::string& id) {
  return o.at("id")==id && o.at("world_contains").get<bool>() &&
    near(o.at("center_m").get<V>(),center) && std::abs(o.at("radius_m").get<double>()-radius)<eps;
}
json examine_path(Checks& check,MoveItFixture& f,const json& path,const V& start,const V& goal,
                  const std::string& label,const V& center={},double radius=0) {
  const auto& points=path.at("points");
  bool count=points.is_array() && points.size()>=2 && points.size()<=1000;
  check.test(label+" solved with finite waypoint count",path.at("solved").get<bool>() && count);
  if(!count) return {{"point_count",points.size()},{"error_code",path.at("error_code")}};
  check.test(label+" endpoint constraints",near(points.front().at("q").get<V>(),start,.002) && near(points.back().at("q").get<V>(),goal,.002));
  bool valid=true,free=true,segments=true;double length=0,clearance=INFINITY,detour=0;
  V line_start(start.begin(),start.begin()+3),line_goal(goal.begin(),goal.begin()+3);
  for(size_t i=0;i<points.size();++i){
    valid=valid_row(points[i]) && valid;
    V q=points[i].at("q").get<V>();
    auto col=f.collision(q);
    free=!col.at("collision").get<bool>() && valid_row(col.at("state")) && free;
    V p=points[i].at("tool_position_m").get<V>();
    detour=std::max(detour,segment_distance(line_start,line_goal,p));
    if(i){V prev=points[i-1].at("tool_position_m").get<V>();length+=distance(prev,p);
      if(!center.empty()){double d=segment_distance(prev,p,center);clearance=std::min(clearance,d-.04-radius);segments=segments && d*d>=(.04+radius)*(.04+radius)-1e-7;}}
  }
  check.test(label+" finite bounded states and independent generated FK",valid);
  check.test(label+" every actual waypoint checked in current FCL scene",free);
  if(!center.empty()) check.test(label+" analytic swept tool-sphere segment clearance",segments);
  json m={{"point_count",points.size()},{"translation_length_m",length},{"max_transverse_detour_m",detour},{"error_code",path.at("error_code")}};
  if(std::isfinite(clearance))m["minimum_segment_clearance_m"]=clearance;
  return m;
}
}

json native_case(const json& input, MoveItFixture& f) {
  using namespace finite_moveit;
  Checks ck;json measurements=json::object();
  try {
    const std::string id=input.at("id").get<std::string>();
    const json& r=input.at("request");
    if(id=="F38_float_arm_variable_map" || id=="F38_unknown_variable_reject") {
      auto m=f.metadata();auto vars=m.at("variables").get<std::vector<std::string>>();
      auto group=m.at("group_variables").get<std::vector<std::string>>();
      const std::vector<std::string> arm={"slide_x","slide_y","slide_z","roll","pitch","yaw"};
      ck.test("requested arm group matches six declared axes",r.at("group")=="arm" && group==arm);
      ck.test("13 distinct variables and initialized actual plugins",vars.size()==13 && std::set<std::string>(vars.begin(),vars.end()).size()==13 && m.at("KDL_loaded").get<bool>() && m.at("OMPL_loaded").get<bool>() && m.at("KDL_joint_names").get<std::vector<std::string>>()==group);
      bool map_ok=true, floating=false;std::set<int> indices;
      for(const auto& j:m.at("joints")) {
        auto name=j.at("name").get<std::string>();auto jvars=j.at("variables").get<std::vector<std::string>>();
        if(name=="floating_base") floating=jvars.size()==7;
        if(std::find(arm.begin(),arm.end(),name)!=arm.end()){
          int index=j.at("first_variable_index").get<int>();
          map_ok=map_ok && jvars.size()==1 && jvars.front()==name && index>=0 && size_t(index)<vars.size() && vars[size_t(index)]==name;
          indices.insert(index);
        }
      }
      ck.test("floating base seven plus six distinct named variable indices",floating && map_ok && indices.size()==6);
      bool known=true;
      for(auto it=r.at("state").begin();it!=r.at("state").end();++it)
        known=known && std::find(vars.begin(),vars.end(),it.key())!=vars.end() && std::find(group.begin(),group.end(),it.key())!=group.end();
      if(!known) {
        ck.test("unknown variable rejected before any state assignment",id=="F38_unknown_variable_reject");
        measurements["error_code"]="UNKNOWN_VARIABLE";
      } else {
        ck.test("positive state has all six variables",r.at("state").size()==6 && id=="F38_float_arm_variable_map");
        V q;for(const auto& name:group)q.push_back(r.at("state").at(name).get<double>());
        auto t=f.transform(q);
        ck.test("real default-base FK matches mapped translation and rotation",bounds(q) && valid_row(t.at("state")) && near(t.at("position_m").get<V>(),V(q.begin(),q.begin()+3)) && (rotation(t.at("rotation"))-fk_rotation(q)).norm()<eps);
        measurements["tool_position_m"]=t.at("position_m");measurements["group_variables"]=group;
      }
    } else if(id=="F39_world_object_identity_free" || id=="F39_changed_pose_collision") {
      V q=r.at("state").get<V>(), center=r.at("sphere_center").get<V>();double radius=r.at("sphere_radius").get<double>();
      auto o=f.sphere(r.at("object_id").get<std::string>(),center,radius);
      ck.test("actual current sphere retains exact object identity pose radius",object_matches(o,center,radius,r.at("object_id").get<std::string>()));
      auto c=f.collision(q);double separation=distance(V(q.begin(),q.begin()+3),center)-.04-radius;
      ck.test("actual collision queried at requested bounded FK state",valid_row(c.at("state")) && near(c.at("state").at("q").get<V>(),q));
      ck.test("FCL classification matches independently computed sphere separation",std::abs(separation)>eps && c.at("collision").get<bool>()==(separation<0));
      measurements={{"collision",c.at("collision")},{"sphere_surface_separation_m",separation},{"world_object",o}};
    } else if(id=="F40_obstacle_replan") {
      V start=r.at("start").get<V>(),goal=r.at("goal").get<V>(),center=r.at("obstacle_center").get<V>();
      double radius=r.at("obstacle_radius").get<double>(),seconds=r.at("planning_time_s").get<double>();
      f.clear_world();auto baseline=f.plan(start,goal,seconds);
      auto bm=examine_path(ck,f,baseline,start,goal,"baseline");
      auto o=f.sphere("generated_obstacle",center,radius);ck.test("replan scene contains requested obstacle",object_matches(o,center,radius,"generated_obstacle"));
      auto replan=f.plan(start,goal,seconds);
      auto rm=examine_path(ck,f,replan,start,goal,"obstacle",center,radius);
      ck.test("new actual trajectory differs and has positive obstacle detour",baseline.at("points")!=replan.at("points") && rm.value("max_transverse_detour_m",0.)>.04+radius-1e-5);
      measurements={{"baseline",bm},{"obstacle",rm}};
    } else if(id=="F40_colliding_goal_no_plan") {
      V start=r.at("start").get<V>(),goal=r.at("goal").get<V>(),center=r.at("obstacle_center").get<V>();double radius=r.at("obstacle_radius").get<double>();
      auto o=f.sphere("generated_obstacle",center,radius);ck.test("negative planner scene object matches",object_matches(o,center,radius,"generated_obstacle"));
      auto col=f.collision(goal);ck.test("actual goal is in collision",col.at("collision").get<bool>() && valid_row(col.at("state")));
      auto p=f.plan(start,goal,r.at("planning_time_s").get<double>());
      ck.test("real planner rejects colliding goal without trajectory",!p.at("solved").get<bool>() && p.at("points").empty() && p.at("error_code").get<int>()!=1);
      measurements={{"error_code",p.at("error_code")},{"point_count",p.at("points").size()}};
    } else if(id=="F41_link_contact_frame" || id=="F41_unknown_contact_frame") {
      V q=r.at("state").get<V>();auto t=f.transform(q,r.at("link").get<std::string>());
      if(t.contains("error")) {
        ck.test("real model rejects unknown frame without identity fallback",id=="F41_unknown_contact_frame" && t.at("error")=="UNKNOWN_FRAME");
        measurements["error_code"]=t.at("error");
      } else {
        auto R=rotation(t.at("rotation"));V pos=t.at("position_m").get<V>(),lp=r.at("local_contact_point_m").get<V>(),ln=r.at("local_normal").get<V>();
        ck.test("real link rotation proper and matches generated serial FK",id=="F41_link_contact_frame" && proper(R) && valid_row(t.at("state")) && (R-fk_rotation(q)).norm()<eps);
        if(!finite(pos,3)||!finite(lp,3)||!finite(ln,3))throw std::runtime_error("CONTACT_VECTOR_NONFINITE");
        Eigen::Vector3d local(lp[0],lp[1],lp[2]),normal(ln[0],ln[1],ln[2]),origin(pos[0],pos[1],pos[2]);
        auto lever=(R*local).eval();auto world=(origin+lever).eval();auto nw=(R*normal).eval();
        ck.test("rotated point normal and lever preserve local geometry",world.allFinite() && nw.allFinite() && (R.transpose()*lever-local).norm()<eps && std::abs(nw.norm()-normal.norm())<eps && std::abs(nw.norm()-1)<eps);
        measurements={{"contact_world_m",{world[0],world[1],world[2]}},{"normal_world",{nw[0],nw[1],nw[2]}},{"lever_arm_world_m",{lever[0],lever[1],lever[2]}}};
      }
    } else if(id=="F42_cartesian_path_finite" || id=="F42_cartesian_out_of_bounds") {
      V start=r.at("start").get<V>(),target=r.at("target_m").get<V>();f.clear_world();
      auto p=f.cartesian(start,target,r.at("step_m").get<double>());double fraction=p.at("fraction").get<double>();const auto& pts=p.at("points");
      ck.test("finite Cartesian fraction and bounded sample count",std::isfinite(fraction) && fraction>=0 && fraction<=1 && pts.size()<=1000);
      bool valid=true;double maxjump=0,max_translation_step=0;
      for(size_t i=0;i<pts.size();++i){valid=valid_row(pts[i]) && valid;
        if(i){V a=pts[i-1].at("q").get<V>(),b=pts[i].at("q").get<V>();for(size_t j=0;j<6;++j)maxjump=std::max(maxjump,std::abs(a[j]-b[j]));max_translation_step=std::max(max_translation_step,distance(pts[i-1].at("tool_position_m").get<V>(),pts[i].at("tool_position_m").get<V>()));}}
      ck.test("all accepted KDL samples finite bounded and FK consistent",valid);
      if(id=="F42_cartesian_path_finite") {
        ck.test("full path with requested initial state and terminal tool pose",fraction>=1-1e-8 && pts.size()>1 && near(pts.front().at("q").get<V>(),start,.002) && near(pts.back().at("tool_position_m").get<V>(),target,.002));
        ck.test("measured adjacent coordinate jumps bounded",maxjump<=.05);
      } else {
        ck.test("out of bounds target cannot yield full accepted trajectory",fraction<1-1e-6 && (pts.empty() || !near(pts.back().at("tool_position_m").get<V>(),target,.002)));
      }
      measurements={{"fraction",fraction},{"point_count",pts.size()},{"max_coordinate_jump",maxjump},{"max_translation_step_m",max_translation_step}};
      if(!pts.empty()){measurements["first_tool_position_m"]=pts.front().at("tool_position_m");measurements["last_tool_position_m"]=pts.back().at("tool_position_m");}
    } else ck.test("recognized finite case",false);
  } catch(const std::exception& e) {
    ck.test("finite test completed without unexpected exception",false);measurements["exception"]=e.what();
  }
  return ck.result(measurements);
}

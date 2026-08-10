#include "arith/ir_visitor_with_analyzer.h"
#include "../op/ascend.h"
#include <tvm/tir/transform.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/relay/expr.h>

// FIXME
// JUST FOR DEBUG
// EXAMPLE: hook<decltype(x)> h;
template<typename T>
class hook;


namespace tvm {
namespace tl {

using namespace tir;
using arith::IRVisitorWithAnalyzer;

#include <sstream>
#include <tvm/ir/expr.h>

template<typename Arr>
std::string to_string(const Arr& arr) {
  std::ostringstream os;
  os << "[";
  for (int i = 0; i < arr.size(); ++i) {
    if (i) os << ", ";
    os << arr[i];
  }
  os << "]";
  return os.str();
}

void read_args_in_string(const std::string& str,std::vector<std::string> &ret) {
  // "tl::ascend::gemm_v0<half, float, 128, 256, 64, false, false>"
  size_t start = str.find('<') + 1;
  size_t end = str.find_last_of('>');
  std::string content = str.substr(start, end - start);
  std::stringstream ss(content);
  std::string item;
  while (std::getline(ss, item, ',')) {
    item.erase(0, item.find_first_not_of(" "));
    item.erase(item.find_last_not_of(" ") + 1);
    ret.push_back(item);
  }
  return;
}

class DSL_Op {
public:
  virtual void to_json(std::string& out) const = 0;
};

class DSL_Tile_Op : public DSL_Op{
public:
  /*
  操作的IR
  :param level: str           比如TileOp还是TensorOp
  :param op_type: str         比如ADD
  :param inputs: list[str]    变量tensor的名称即可，具体的tensor内容有其他地方表达
  :param outputs: list[str]   变量tensor的名称即可，具体的tensor内容有其他地方表达
  :param config: dict         其他可能要拓展的配置, json style
  */
  std::string level = "TileOp";
  std::string op_type;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  std::unordered_map<std::string, std::string> config;

  DSL_Tile_Op(std::string level, 
       std::string op_type, 
       std::vector<std::string> inputs, 
       std::vector<std::string> outputs, 
       std::unordered_map<std::string, std::string> config)
  : level(std::move(level)), 
    op_type(std::move(op_type)), 
    inputs(std::move(inputs)), 
    outputs(std::move(outputs)), 
    config(std::move(config)) {}

  void to_json(std::string& out) const {
    out += "{";
    out += "\"level\":\"" + level + "\",";
    out += "\"op_type\":\"" + op_type + "\",";
    out += "\"inputs\":[";
    for (size_t i = 0; i < inputs.size(); ++i) {
        out += "\"" + inputs[i] + "\"";
        if (i < inputs.size() - 1) out += ",";
    }
    out += "],";
    out += "\"outputs\":[";
    for (size_t i = 0; i < outputs.size(); ++i) {
        out += "\"" + outputs[i] + "\"";
        if (i < outputs.size() - 1) out += ",";
    }
    out += "],";
    out += "\"config\":{";
    auto it = config.begin();
    while (it != config.end()) {
        out += "\"" + it->first + "\":\"" + it->second + "\"";
        if (++it != config.end()) out += ",";
    }
    out += "}}";
  }
};

class DSL_Tensor : public DSL_Op{
public:
  /*
  张量IR
  :param name:    张量的名称，唯一不重复
  :param shape:   张量的形状，list格式，比如[128, 128]
  :param dtype:   张量的数据类型，str格式，比如FLOAT32
  :param mem_loc: 张量所处的存储层级，str格式，比如GM
  */
  std::string name;
  std::string dtype;
  std::string mem_loc;
  std::vector<std::string> shape;
  DSL_Tensor() = default;
  DSL_Tensor(std::string name, 
            std::string dtype, 
            std::string mem_loc, 
            std::vector<std::string> shape):
  name(std::move(name)), 
  dtype(std::move(dtype)), 
  mem_loc(std::move(mem_loc)), 
  shape(std::move(shape)) {}

  void to_json(std::string& out) const {
    out += "{";
    out += "\"name\":\"" + name + "\",";
    out += "\"dtype\":\"" + dtype + "\",";
    out += "\"mem_loc\":\"" + mem_loc + "\",";
    out += "\"shape\":[";
    for (size_t i = 0; i < shape.size(); ++i) {
        out += "\"" + shape[i] + "\""; 
        if (i < shape.size() - 1) out += ",";
    }
    out += "]}";
  }

};

class DSL_Var {
public:
  std::string name;
  std::string value;
  DSL_Var() = default;
  DSL_Var(std::string name, std::string value)
  : name(std::move(name)), value(std::move(value)) {}
};

class DSL_For : public DSL_Op{
public:
  std::string level = "ForOp";
  std::string var;
  std::string min;
  std::string extent;
  std::vector<std::shared_ptr<DSL_Op>> op_list;

  DSL_For() = default;
  DSL_For(std::string var, 
          std::string extent)
  : var(std::move(var)), 
    extent(std::move(extent)){
      min = "0";
    }
  DSL_For(std::string var, 
          std::string min, 
          std::string extent,
          std::vector<std::shared_ptr<DSL_Op>> op_list)
  : var(std::move(var)), 
    min(std::move(min)), 
    extent(std::move(extent)), 
    op_list(std::move(op_list)){}

  void to_json(std::string& out) const {
    out += "{";
    out += "\"level\":\"" + level + "\",";
    out += "\"var\":\"" + var + "\",";
    out += "\"min\":\"" + min + "\",";
    out += "\"extent\":\"" + extent + "\",";
    out += "\"op_list\":[\n";
    for (size_t i = 0; i < op_list.size(); ++i) {
        out += "        ";
        op_list[i]->to_json(out);
        if (i < op_list.size() - 1) out += ",\n";
        else out += "\n";
    }
    out += "    ]}";
  }
};

class DSL_Task{
public:
  std::vector<std::string> vars;
  std::vector<std::string> ranges;
  std::vector<std::shared_ptr<DSL_Op>> op_list;

  DSL_Task() = default;
  DSL_Task(std::vector<std::string> vars, 
          std::vector<std::string> ranges, 
          std::vector<std::shared_ptr<DSL_Op>> op_list)
  : vars(std::move(vars)), 
    ranges(std::move(ranges)), 
    op_list(std::move(op_list)){}
  void to_json(std::string& out) const {
    out += "\"vars\":[";
    for (size_t i = 0; i < vars.size(); ++i) {
        out += "\"" + vars[i] + "\"";
        if (i < vars.size() - 1) out += ",";
    }
    out += "],";
    out += "\"ranges\":[";
    for (size_t i = 0; i < ranges.size(); ++i) {
        out += "\"" + ranges[i] + "\"";
        if (i < ranges.size() - 1) out += ",";
    }
    out += "]";
  }
};


class TileGraphCollector : public IRVisitorWithAnalyzer {
public:
  static void Collect(const Stmt &stmt) {
    TileGraphCollector collector;
    collector(stmt);
    collector.dump_kernel_();
  }

  TileGraphCollector() = default;

private:
  void insert_buffer_(const tvm::tir::Buffer &buffer) {
    std::string name = buffer->name;
    if (tensor_table.find(name) == tensor_table.end()) {
      std::string dtype;
      std::string mem_loc;
      std::vector<std::string> shape;
      if (buffer->dtype.bits() == 16)
        dtype = "fp16";
      else if (buffer->dtype.bits() == 32)
        dtype = "fp32";
      else
        dtype = "UNKNOWN";
      mem_loc = buffer.scope();
      for (auto dim : buffer->shape) {
        shape.push_back(std::to_string(dim.as<tvm::tir::IntImmNode>()->value));
      }
      tensor_table[name] = DSL_Tensor(name, dtype, mem_loc, shape);
    }
  }

  void VisitStmt_(const BlockNode* op) {
    for (auto buf_region : op->reads) {
      insert_buffer_(buf_region->buffer);
    }
    for (auto buf_region : op->writes) {
      insert_buffer_(buf_region->buffer);
    }
    for (auto buf : op->alloc_buffers) {
      insert_buffer_(buf);
    }
    for (auto match_buf_region : op->match_buffers) {
      insert_buffer_(match_buf_region->buffer);
      insert_buffer_(match_buf_region->source->buffer);
    }
    IRVisitorWithAnalyzer::VisitStmt_(op);
  }

  void VisitStmt_(const ForNode *op) final {
    PrimExpr extent = analyzer_.Simplify(op->extent);
    if (const IntImmNode* imm = extent.as<IntImmNode>()) {
      op_list.push_back(std::make_shared<DSL_For>(op->loop_var->name_hint, std::to_string(imm->value)));
      size_t for_idx = op_list.size() -1;
      IRVisitorWithAnalyzer::VisitStmt_(op);
      for (size_t idx = for_idx +1; idx < op_list.size(); ++idx) {
        std::static_pointer_cast<DSL_For>(op_list[for_idx])->op_list.push_back(op_list[idx]);
      }
      op_list.erase(op_list.begin() + for_idx +1, op_list.end());
    } else {
      LOG(FATAL) << "Find dynamic loop" << extent;
    }
  }

  void VisitStmt_(const IfThenElseNode *op) final {
    LOG(FATAL) << "Find IF";
    IRVisitorWithAnalyzer::VisitStmt_(op);
  }

  void VisitStmt_(const LetStmtNode* op) final {
    var_list.push_back(DSL_Var(op->var->name_hint, tvm::relay::PrettyPrint(analyzer_.Simplify(op->value))));
    IRVisitorWithAnalyzer::VisitStmt_(op);
  }

  void VisitStmt_(const AttrStmtNode* op) {
    const tvm::tir::IterVarNode* iter_var = op->node.as<tvm::tir::IterVarNode>();
    if (iter_var != nullptr) {
      task.vars.push_back(iter_var->var->name_hint);
      PrimExpr extent = analyzer_.Simplify(op->value);
      task.ranges.push_back(tvm::relay::PrettyPrint(extent));
    }
    IRVisitorWithAnalyzer::VisitStmt_(op);
  }

  void VisitExpr_(const CallNode *op) { 
    if (op->op.same_as(Op::Get("tl.ascend_copy"))) {
      std::unordered_map<std::string, std::string> config;
      auto src_region = op->args[0].as<CallNode>();
      auto dst_region = op->args[1].as<CallNode>();
      auto src_buffer = src_region->args[0].as<BufferLoadNode>()->buffer;
      auto dst_buffer = dst_region->args[0].as<BufferLoadNode>()->buffer;
      auto src_indices = src_region->args[0].as<BufferLoadNode>()->indices;
      auto dst_indices = dst_region->args[0].as<BufferLoadNode>()->indices;
      insert_buffer_(src_buffer);
      insert_buffer_(dst_buffer);
      config["src_indices"] = to_string(src_indices);
      config["dst_indices"] = to_string(dst_indices);
      op_list.push_back(std::make_shared<DSL_Tile_Op>("TileOp", "copy", 
                                        std::vector<std::string>{src_buffer->data.as<VarNode>()->name_hint}, 
                                        std::vector<std::string>{dst_buffer->data.as<VarNode>()->name_hint}, 
                                        config));
    } else if (op->op.same_as(Op::Get("tl.ascend_pipe_barrier"))) {
      op_list.push_back(std::make_shared<DSL_Tile_Op>("TileOp", "pipebarrier", 
                                        std::vector<std::string>(), 
                                        std::vector<std::string>(), 
                                        std::unordered_map<std::string, std::string>()));
    } else {
      std::vector<std::string> binary_ops_for_2tiles = {"add", "sub", "mul", "div", "max", "min", "and", "or"};
      std::vector<std::string> binary_ops_for_tile_scaler = {"adds", "subs", "muls", "divs", "maxs", "mins", "ands", "ors"};
      std::vector<std::string> unary_ops = {"exp", "ln", "abs", "reciprocal", "sqrt", "rsqrt", "relu", "not"};
      std::string op_type = tvm::relay::PrettyPrint(op->op);
      std::vector<std::string> inputs;
      std::vector<std::string> outputs;
      std::vector<std::string> input_offsets;
      std::vector<std::string> output_offsets;
      std::unordered_map<std::string, std::string> config;
      
      if (op_type.find("Flag") != std::string::npos) {
        inputs.push_back(tvm::relay::PrettyPrint(op->args[1]));
        if (op_type.find("Set") != std::string::npos) {
          op_list.push_back(std::make_shared<DSL_Tile_Op>("TileOp", "crosscoresetflag", inputs, outputs, config));
        } else if (op_type.find("Wait") != std::string::npos) {
          op_list.push_back(std::make_shared<DSL_Tile_Op>("TileOp", "crosscorewaitflag", inputs, outputs, config));
        } else {
          LOG(FATAL) << "Unknown Flag Op Type: " << op_type;
        }
      } else if (op_type.find("gemm_v0") != std::string::npos) {
        std::vector<std::string> op_str_args;
        auto a_ptr = op->args[1].as<CallNode>();
        auto b_ptr = op->args[2].as<CallNode>();
        auto c_ptr = op->args[3].as<CallNode>();
        auto a_name = a_ptr->args[1].as<tvm::tir::VarNode>()->name_hint;
        auto b_name = b_ptr->args[1].as<tvm::tir::VarNode>()->name_hint;
        auto c_name = c_ptr->args[1].as<tvm::tir::VarNode>()->name_hint;
        auto a_offset = tvm::relay::PrettyPrint(a_ptr->args[2]);
        auto b_offset = tvm::relay::PrettyPrint(b_ptr->args[2]);
        auto c_offset = tvm::relay::PrettyPrint(c_ptr->args[2]);
        inputs.insert(inputs.end(), {a_name, b_name});
        input_offsets.insert(input_offsets.end(), {a_offset, b_offset});
        outputs.push_back(c_name);
        output_offsets.push_back(c_offset);
        read_args_in_string(tvm::relay::PrettyPrint(op->args[0]), op_str_args);
        config["dtype"] = op_str_args[0];
        config["accum_dtype"] = op_str_args[1];
        config["block_M"] = op_str_args[2];
        config["block_N"] = op_str_args[3];
        config["block_K"] = op_str_args[4];
        config["transpose_A"] = op_str_args[5];
        config["transpose_B"] = op_str_args[6];
        config["clear_accum"] = tvm::relay::PrettyPrint(op->args[4]);
        // FIXME error: no matching function for call to ‘to_string(std::vector<std::__cxx11::basic_string<char> >&)’
        config["input_offsets"] = to_string(input_offsets);
        config["input_offsets"] = to_string(input_offsets);
        config["output_offsets"] = to_string(output_offsets);
        op_list.push_back(std::make_shared<DSL_Tile_Op>("TileOp", "gemm_v0", inputs, outputs, config));

      } else if (std::find(binary_ops_for_2tiles.begin(), binary_ops_for_2tiles.end(), op_type.substr(10)) != binary_ops_for_2tiles.end()) {
        auto dst_ptr = op->args[0].as<CallNode>();
        auto src0_ptr = op->args[1].as<CallNode>();
        auto src1_ptr = op->args[2].as<CallNode>();
        auto dst_name = dst_ptr->args[1].as<tvm::tir::VarNode>()->name_hint;
        auto src0_name = src0_ptr->args[1].as<tvm::tir::VarNode>()->name_hint;
        auto src1_name = src1_ptr->args[1].as<tvm::tir::VarNode>()->name_hint;
        auto src0_offset = tvm::relay::PrettyPrint(src0_ptr->args[2]);
        auto src1_offset = tvm::relay::PrettyPrint(src1_ptr->args[2]);
        auto dst_offset = tvm::relay::PrettyPrint(dst_ptr->args[2]);
        if (op->args.size() >= 5) {
          config["indices_1"] = tvm::relay::PrettyPrint(op->args[3]);
          config["size_0"] = tvm::relay::PrettyPrint(op->args[4]);
        } else {
          config["size_0"] = tvm::relay::PrettyPrint(op->args[3]);
        }
        inputs.insert(inputs.end(), {src0_name, src1_name});
        outputs.push_back(dst_name);
        input_offsets.insert(input_offsets.end(), {src0_offset, src1_offset});
        output_offsets.push_back(dst_offset);
        config["input_offsets"] = to_string(input_offsets);
        config["output_offsets"] = to_string(output_offsets);
        op_list.push_back(std::make_shared<DSL_Tile_Op>("TileOp", op_type.substr(10), inputs, outputs, config));

      } else if (std::find(binary_ops_for_tile_scaler.begin(), binary_ops_for_tile_scaler.end(), op_type.substr(10)) != binary_ops_for_tile_scaler.end()) {
        auto dst_ptr = op->args[0].as<CallNode>();
        auto src0_ptr = op->args[1].as<CallNode>();
        auto dst_name = dst_ptr->args[1].as<tvm::tir::VarNode>()->name_hint;
        auto src0_name = src0_ptr->args[1].as<tvm::tir::VarNode>()->name_hint;
        auto dst_offset = tvm::relay::PrettyPrint(dst_ptr->args[2]);
        auto src0_offset = tvm::relay::PrettyPrint(src0_ptr->args[2]);
        inputs.push_back(src0_name);
        outputs.push_back(dst_name);
        input_offsets.push_back(src0_offset);
        output_offsets.push_back(dst_offset);
        auto src1_ptr = op->args[2].as<CallNode>();
        if (src1_ptr == nullptr) {
          inputs.push_back(tvm::relay::PrettyPrint(analyzer_.Simplify(op->args[2])));
        } else {
          inputs.push_back(src1_ptr->args[1].as<tvm::tir::VarNode>()->name_hint);
        }
        config["size_0"] = tvm::relay::PrettyPrint(analyzer_.Simplify(op->args[3]));
        config["input_offsets"] = to_string(input_offsets);
        config["output_offsets"] = to_string(output_offsets);
        op_list.push_back(std::make_shared<DSL_Tile_Op>("TileOp", op_type.substr(10), inputs, outputs, config));

      } else if (std::find(unary_ops.begin(), unary_ops.end(), op_type.substr(10)) != unary_ops.end()) {
        auto dst_ptr = op->args[0].as<CallNode>();
        auto src0_ptr = op->args[1].as<CallNode>();
        auto dst_name = dst_ptr->args[1].as<tvm::tir::VarNode>()->name_hint;
        auto src0_name = src0_ptr->args[1].as<tvm::tir::VarNode>()->name_hint;
        auto dst_offset = tvm::relay::PrettyPrint(dst_ptr->args[2]);
        auto src0_offset = tvm::relay::PrettyPrint(src0_ptr->args[2]);
        config["size_0"] = tvm::relay::PrettyPrint(analyzer_.Simplify(op->args[2]));
        inputs.push_back(src0_name);
        outputs.push_back(dst_name);
        input_offsets.push_back(src0_offset);
        output_offsets.push_back(dst_offset);
        config["input_offsets"] = to_string(input_offsets);
        config["output_offsets"] = to_string(output_offsets);
        op_list.push_back(std::make_shared<DSL_Tile_Op>("TileOp", op_type.substr(10), inputs, outputs, config));
      
      } else if (op_type.find("reduce") != std::string::npos) {
        std::vector<std::string> op_str_args;
        std::string shape_str = "[";
        auto out_ptr = op->args[1].as<CallNode>();
        auto buffer_ptr = op->args[2].as<CallNode>();
        auto tmp_ptr = op->args[3].as<CallNode>();
        auto buffer_name = buffer_ptr->args[1].as<tvm::tir::VarNode>()->name_hint;
        auto out_name = out_ptr->args[1].as<tvm::tir::VarNode>()->name_hint;
        inputs.push_back(buffer_name);
        outputs.push_back(out_name);
        input_offsets.push_back(tvm::relay::PrettyPrint(buffer_ptr->args[2]));
        output_offsets.push_back(tvm::relay::PrettyPrint(out_ptr->args[2]));
        config["input_offsets"] = to_string(input_offsets);
        config["output_offsets"] = to_string(output_offsets);
        read_args_in_string(tvm::relay::PrettyPrint(op->args[0]), op_str_args);
        config["dtype"] = op_str_args[0];
        for (size_t i = 1; i < op_str_args.size() - 1; ++i) {
          shape_str += op_str_args[i];
          if (i < op_str_args.size() - 2) shape_str += ", ";
        }
        shape_str += "]";
        config["shape"] = shape_str;
        config["pattern"] = op_str_args.back();
        config["tmp"] = tmp_ptr->args[1].as<tvm::tir::VarNode>()->name_hint;
        op_list.push_back(std::make_shared<DSL_Tile_Op>("TileOp", std::string(tvm::relay::PrettyPrint(op->args[0])).substr(1, 10), inputs, outputs, config));

      } else if (op_type.find("fill") != std::string::npos) {
        std::vector<std::string> op_str_args;
        auto dst_ptr = op->args[1].as<CallNode>();
        inputs.push_back(tvm::relay::PrettyPrint(analyzer_.Simplify(op->args[2])));
        outputs.push_back(dst_ptr->args[1].as<tvm::tir::VarNode>()->name_hint);
        output_offsets.push_back(tvm::relay::PrettyPrint(dst_ptr->args[2]));
        read_args_in_string(tvm::relay::PrettyPrint(op->args[0]), op_str_args);
        config["size_0"] = tvm::relay::PrettyPrint(analyzer_.Simplify(op->args[3]));
        config["dtype"] = op_str_args[0];
        config["output_offsets"] = to_string(output_offsets);
        op_list.push_back(std::make_shared<DSL_Tile_Op>("TileOp", "fill", inputs, outputs, config));
      } else {
        LOG(INFO) << "Unknown Extern Op Type: " << op_type;
      } 
    }
  }

  void dump_kernel_(const std::string& filename = "ir.json") {
    std::string json_str = "{\n";
    json_str += "  \"task\": {\n";
    task.to_json(json_str);
    json_str += "  },\n";
    json_str += "  \"vars\": [\n";
    for (size_t i = 0; i < var_list.size(); ++i) {
        json_str += "    {\"name\": \"" + var_list[i].name + "\", \"value\": \"" + var_list[i].value + "\"}";
        if (i < var_list.size() - 1) json_str += ",\n";
        else json_str += "\n";
    }
    json_str += "  ],\n";
    json_str += "  \"ops\": [\n";
    for (size_t i = 0; i < op_list.size(); ++i) {
        json_str += "    ";
        if (op_list[i]) {
            op_list[i]->to_json(json_str);
        }
        if (i < op_list.size() - 1) json_str += ",\n";
        else json_str += "\n";
    }
    json_str += "  ],\n";
    json_str += "  \"tensors\": {\n";
    auto it = tensor_table.begin();
    while (it != tensor_table.end()) {
        json_str += "    \"" + it->first + "\": ";
        it->second.to_json(json_str);
        if (++it != tensor_table.end()) json_str += ",\n";
        else json_str += "\n";
    }
    json_str += "  }\n}";
    std::ofstream ofs(filename);
    if (ofs.is_open()) {
        ofs << json_str;
        ofs.close();
        LOG(INFO) << "Kernel IR dumped to " << filename;
    }
  }

  std::unordered_map<std::string, DSL_Tensor> tensor_table;
  std::vector<std::shared_ptr<DSL_Op>> op_list;
  std::vector<DSL_Var> var_list;
  DSL_Task task;
};

void DumpTileGraph(const tir::PrimFunc& func) {
  LOG(INFO) << func;
  TileGraphCollector::Collect(func->body);
}

TVM_REGISTER_GLOBAL("tl.analysis.DumpTileGraph").set_body_typed(DumpTileGraph);

} // namespace tl
} // namespace tvm

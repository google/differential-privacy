//
// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

// Tool for running a query with ZetaSQL. Supports reading from a csv file.

#include <math.h>

#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "google/protobuf/descriptor.h"
#include "zetasql/public/analyzer_options.h"
#include "zetasql/public/builtin_function_options.h"
#include "zetasql/public/catalog.h"
#include "zetasql/public/language_options.h"
#include "zetasql/public/options.pb.h"
#include "zetasql/public/simple_catalog.h"
#include "zetasql/public/type.pb.h"
#include "zetasql/public/value.h"
#include "zetasql/resolved_ast/resolved_ast.h"
#include "zetasql/resolved_ast/resolved_ast_visitor.h"
#include "zetasql/resolved_ast/resolved_node.h"
#include "zetasql/resolved_ast/resolved_node_kind.pb.h"
#include "zetasql/tools/execute_query/execute_query_tool.h"
#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "base/status_macros.h"
#include "absl/flags/parse.h"

ABSL_FLAG(std::string, data_set, "",
          "A CSV file containing the data to be queried, whose std::string-typed "
          "column names are determined from the first (header) row. The data "
          "is loaded into a table with the same name as the file (without the "
          ".csv extension, if it exists).");

ABSL_FLAG(
    std::string, userid_col, "",
    "A std::string matching the name of the column in the  containing the user IDs, "
    "to be used in anonoymization queries.");

// Verifies anonymization parameters to be within valid bounds
class VerifyAnonymizationParametersVisitor
    : public zetasql::ResolvedASTVisitor {
 public:
  // We only need a special visitor function for AnonymizationAggregateScans.
  absl::Status VisitResolvedAnonymizedAggregateScan(
      const zetasql::ResolvedAnonymizedAggregateScan* node) override {
    bool epsilon_provided = false;
    bool delta_provided = false;
    bool kappa_provided = false;

    for (auto const& anon_option : node->anonymization_option_list()) {
      // Extract the anonymization option value as a double
      double anon_option_double;

      std::string name = absl::AsciiStrToUpper(anon_option->name());
      const zetasql::ResolvedExpr* anon_option_expr = anon_option->value();

      switch (anon_option_expr->node_kind()) {
        case zetasql::ResolvedNodeKind::RESOLVED_LITERAL: {
          const zetasql::Value anon_option_value =
              anon_option_expr->GetAs<zetasql::ResolvedLiteral>()->value();
          switch (anon_option_value.type_kind()) {
            case zetasql::TypeKind::TYPE_INT64:
              anon_option_double = anon_option_value.int64_value();
              break;
            case zetasql::TypeKind::TYPE_DOUBLE:
              anon_option_double = anon_option_value.double_value();
              break;
            default:  // Unexpected anon_option_value.type_kind()
              return absl::InternalError(absl::StrCat(
                  "Anonymization option ", name,
                  " is expected to be parsed as either an INT64 or",
                  " DOUBLE, but is a " <
                      anon_option_value.type()->ShortTypeName(
                          zetasql::PRODUCT_EXTERNAL),
                  "."));
              break;
          }
          break;
        }
        default: {  // Unexpected anon_option_expr->node_kind()
          return absl::InvalidArgumentError(absl::StrCat(
              "The value of anonymization option", name, " cannot be ",
              "interpreted, since it is not a literal, but is a ",
              anon_option_expr->node_kind_string(), "."));
          break;
        }
      }

      // Return an error if any anonymization parameter is not positive.
      if (anon_option_double <= 0) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Anonymization option ", name, " must be positive, but is ",
            anon_option_double, "."));
      }

      if (absl::EqualsIgnoreCase(name, "epsilon")) {
        epsilon_provided = true;
      }

      if (absl::EqualsIgnoreCase(name, "delta")) {
        delta_provided = true;
        // Return an error if delta is provided and is larger than 1.
        if (anon_option_double > 1) {
          return absl::InvalidArgumentError(absl::StrCat(
              "Anonymization option ", name,
              " must be greater than 0 and less than or equal to 1, but",
              " is ", anon_option_double, "."));
        }
      }

      if (absl::EqualsIgnoreCase(name, "kappa")) {
        kappa_provided = true;
        // Return an error if kappa is specified but is not integer,
        // in case the SQL interpreter did not catch it first.
        double intpart;
        double fraction = modf(anon_option_double, &intpart);
        if (fraction != 0.0) {
          return absl::InvalidArgumentError(absl::StrCat(
              "Anonymization option ", name, " must be an integer, but is ",
              anon_option_double, "."));
        }
      }

      // Return an error k_threshold is specified. Delta should be used instead.
      if (absl::EqualsIgnoreCase(name, "k_threshold")) {
        return absl::InvalidArgumentError(
            "Please use DELTA instead of K_THRESHOLD. DELTA can be"
            " calculated using Theorem 2 of Wilson et al.'s paper on"
            " Differentially Private SQL with Bounded User Contribution"
            " (available at https://arxiv.org/pdf/1909.01917.pdf).");
      }
    }

    // Return an error if epsilon, delta, or kappa are not provided
    if (!(epsilon_provided && delta_provided && kappa_provided)) {
      return absl::InvalidArgumentError(
          "ZetaSQL differential privacy queries must specify EPSILON, "
          " DELTA, and KAPPA in the WITH ANONYMIZATION OPTIONS() clause.");
    }
    return absl::OkStatus();
  }
};

// Returns the file name (without the ".csv", if any) from file_path
static std::string GetCSVFileNameFromPath(const std::string_view file_path) {
  std::vector<std::string> file_path_tokens = absl::StrSplit(file_path, '/');
  std::string_view file_name = file_path_tokens.back();
  absl::ConsumeSuffix(&file_name, ".csv");
  absl::ConsumeSuffix(&file_name, ".CSV");
  return std::string(file_name);
}

// Wrapper to get catalog for a config.
static zetasql::SimpleCatalog& GetCatalogForConfig(
    zetasql::ExecuteQueryConfig& config) {
  return config.mutable_catalog();
}

static absl::Status InitExecuteQueryConfig(
    zetasql::ExecuteQueryConfig& config) {
  config.set_examine_resolved_ast_callback(
      [](const zetasql::ResolvedNode* node) -> absl::Status {
        auto visitor = VerifyAnonymizationParametersVisitor();
        return node->Accept(&visitor);
      });

  RETURN_IF_ERROR(SetToolModeFromFlags(config));

  std::string file_path = absl::GetFlag(FLAGS_data_set);
  std::string table_name = GetCSVFileNameFromPath(file_path);

  ASSIGN_OR_RETURN(std::unique_ptr<zetasql::SimpleTable> table,
                   zetasql::MakeTableFromCsvFile(table_name, file_path));
  const std::string userid_col = absl::GetFlag(FLAGS_userid_col);
  RETURN_IF_ERROR(table->SetAnonymizationInfo({userid_col}));
  config.mutable_analyzer_options().set_enabled_rewrites(
      {zetasql::REWRITE_ANONYMIZATION});
  GetCatalogForConfig(config).AddOwnedTable(std::move(table));

  config.mutable_analyzer_options()
      .mutable_language()
      ->EnableMaximumLanguageFeaturesForDevelopment();
  GetCatalogForConfig(config).AddZetaSQLFunctions(
      config.analyzer_options().language());
  return absl::OkStatus();
}

int main(int argc, char* argv[]) {
  const char kUsage[] =
      "Usage: execute_query --data_set=<path_to_csv_file> "
      "--userid_col=<userid_column_name_in_data_set> <sql_statement>\n";
  std::vector<char*> remaining_args = absl::ParseCommandLine(argc, argv);
  if (argc <= 1) {
    LOG(QFATAL) << kUsage;
  }
  const std::string sql = absl::StrJoin(remaining_args.begin() + 1,
  remaining_args.end(), " ");
  zetasql::ExecuteQueryConfig config;
  absl::Status status = InitExecuteQueryConfig(config);
  if (!status.ok()) {
    std::cout << "ERROR: " << status << std::endl;
    return 1;
  }

  auto writer = zetasql::MakeWriterFromFlags(config, std::cout);
  if (!writer.status().ok()) {
    std::cout << "ERROR: " << writer.status() << std::endl;
    return 1;
  }

  status = ExecuteQuery(sql, config, *writer.value());
  if (!status.ok()) {
    std::cout << "ERROR: " << status << std::endl;
    return 1;
  }

  return 0;
}

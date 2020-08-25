//
// Copyright 2019 Google LLC
// Copyright 2018 ZetaSQL Authors
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

#include "base/logging.h"

#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#ifdef _WIN32
// START PATCH
// This `if` code block conditionally defines and imports windows system
// specific code. This block defines the windows alternatives to the functions
// defined in <unistd.h> which is included in the `else` section below.
#include <Windows.h>
#include <direct.h>  // for: _mkdir
#include <io.h>      // for: access
#include <time.h>    // for: timespec
// By design Windows does not have a `mode` parameter.
#undef mkdir
#define mkdir(path, mode) _mkdir(path)

#ifndef S_ISDIR
#define S_ISDIR(mode) (((mode)&S_IFMT) == S_IFDIR)
#endif

#define W_OK 2
#define F_OK 0
#define CLOCK_REALTIME 0

static int clock_gettime(int, struct timespec *spec)  // C-file part
{
  __int64 wintime;
  GetSystemTimeAsFileTime((FILETIME *)&wintime);
  wintime -= 116444736000000000i64;             // 1jan1601 to 1jan1970
  spec->tv_sec = wintime / 10000000i64;         // seconds
  spec->tv_nsec = wintime % 10000000i64 * 100;  // nano-seconds
  return 0;
}
// END PATCH
#else
#include <unistd.h>
#endif
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <sstream>
#include <string>

#include "absl/base/attributes.h"

namespace differential_privacy {
namespace base {

constexpr char kDefaultDirectory[] = "/tmp/";

namespace {

// The logging directory.
ABSL_CONST_INIT std::string *log_file_directory = nullptr;

// The log filename.
ABSL_CONST_INIT std::string *log_basename = nullptr;

// The VLOG level, only VLOG with level equal to or below this level is logged.
ABSL_CONST_INIT int vlog_level = 0;

const char *GetBasename(const char *file_path) {
  const char *slash = strrchr(file_path, '/');
  return slash ? slash + 1 : file_path;
}

bool set_log_basename(const std::string &filename) {
  if (log_basename || filename.empty()) {
    return false;
  }
  log_basename = new std::string(filename);
  return true;
}

std::string get_log_basename() {
  if (!log_basename || log_basename->empty()) {
    return "zetasql";
  }
  return *log_basename;
}

bool EnsureDirectoryExists(const char *path) {
  struct stat dirStat;
  if (stat(path, &dirStat)) {
    if (errno != ENOENT) {
      return false;
    }
    if (mkdir(path, 0766)) {
      return false;
    }
  } else if (!S_ISDIR(dirStat.st_mode)) {
    return false;
  }
  return true;
}

// Sets the log directory, as specified when initialized. This
// is only set once. Any request to reset it will return false.
//
// log_directory: log file directory.
//
// Returns true if and only if the log directory is set successfully.
bool set_log_directory(const std::string &log_directory) {
  std::string tmp_directory = log_directory;
  if (tmp_directory.empty()) {
    tmp_directory = kDefaultDirectory;
  }
  if (log_file_directory || !EnsureDirectoryExists(tmp_directory.c_str())) {
    return false;
  }
  if (tmp_directory.back() == '/') {
    log_file_directory = new std::string(tmp_directory);
  } else {
    log_file_directory = new std::string(tmp_directory + "/");
  }
  return true;
}

// Sets the verbosity threshold for VLOG. A VLOG command with a level greater
// than this will be ignored.
//
// level: verbosity threshold for VLOG to be set. A VLOG command with
//        level less than or equal to this will be logged.
void set_vlog_level(int level) { vlog_level = level; }

}  // namespace

std::string get_log_directory() {
  if (!log_file_directory) {
    return kDefaultDirectory;
  }
  return *log_file_directory;
}

int get_vlog_level() { return vlog_level; }

bool InitLogging(const char *directory, const char *file_name, int level) {
  set_vlog_level(level);
  std::string log_directory = directory ? std::string(directory) : "";
  if (!set_log_directory(log_directory)) {
    return false;
  }
  const char *binary_name = GetBasename(file_name);
  if (!set_log_basename(binary_name)) {
    return false;
  }
  std::string log_path = get_log_directory() + get_log_basename();
  if (access(log_path.c_str(), F_OK) == 0 &&
      access(log_path.c_str(), W_OK) != 0) {
    return false;
  }
  return true;
}


CheckOpMessageBuilder::CheckOpMessageBuilder(const char *exprtext)
    : stream_(new std::ostringstream) {
  *stream_ << exprtext << " (";
}

CheckOpMessageBuilder::~CheckOpMessageBuilder() { delete stream_; }

std::ostream *CheckOpMessageBuilder::ForVar2() {
  *stream_ << " vs. ";
  return stream_;
}

std::string *CheckOpMessageBuilder::NewString() {  // NOLINT
  *stream_ << ")";
  return new std::string(stream_->str());
}

template <>
void MakeCheckOpValueString(std::ostream *os, const char &v) {
  if (v >= 32 && v <= 126) {
    (*os) << "'" << v << "'";
  } else {
    (*os) << "char value " << static_cast<int16_t>(v);
  }
}

template <>
void MakeCheckOpValueString(std::ostream *os, const signed char &v) {
  if (v >= 32 && v <= 126) {
    (*os) << "'" << v << "'";
  } else {
    (*os) << "signed char value " << static_cast<int16_t>(v);
  }
}

template <>
void MakeCheckOpValueString(std::ostream *os, const unsigned char &v) {
  if (v >= 32 && v <= 126) {
    (*os) << "'" << v << "'";
  } else {
    (*os) << "unsigned char value " << static_cast<uint16_t>(v);
  }
}

template <>
void MakeCheckOpValueString(std::ostream *os, const std::nullptr_t &v) {
  (*os) << "nullptr";
}

namespace logging_internal {

LogMessage::LogMessage(const char *file, int line)
  : LogMessage(file, line, absl::LogSeverity::kInfo) {}

LogMessage::LogMessage(const char *file, int line, const std::string &result)
    : LogMessage(file, line, absl::LogSeverity::kFatal) {
  stream() << "Check failed: " << result << " ";
}

static constexpr const char *LogSeverityNames[4] = {"INFO", "WARNING", "ERROR",
                                                    "FATAL"};

LogMessage::LogMessage(const char *file, int line, absl::LogSeverity severity)
    : severity_(severity) {
  const char *filename = GetBasename(file);

  // Write a prefix into the log message, including local date/time, severity
  // level, filename, and line number.
  struct timespec time_stamp;
  clock_gettime(CLOCK_REALTIME, &time_stamp);

  constexpr int kTimeMessageSize = 22;
  char buffer[kTimeMessageSize];
  strftime(buffer, kTimeMessageSize, "%Y-%m-%d %H:%M:%S  ",
           localtime(&time_stamp.tv_sec));
  stream() << buffer;
  stream() << LogSeverityNames[static_cast<int>(severity)] << "  "
           << filename << " : " << line << " : ";
}

LogMessage::~LogMessage() {
  Flush();
  // if FATAL occurs, abort.
  if (severity_ == absl::LogSeverity::kFatal) {
    abort();
  }
}

void LogMessage::SendToLog(const std::string &message_text) {
  std::string log_path = get_log_directory() + get_log_basename();

  FILE *file = fopen(log_path.c_str(), "ab");
  if (file) {
    if (fprintf(file, "%s", message_text.c_str()) > 0) {
      if (message_text.back() != '\n') {
        fprintf(file, "\n");
      }
    } else {
      fprintf(stderr, "Failed to write to log file : %s! [%s]\n",
              log_path.c_str(), strerror(errno));
    }
    fclose(file);
  } else {
    fprintf(stderr, "Failed to open log file : %s! [%s]\n",
            log_path.c_str(), strerror(errno));
  }
  if (severity_ >= absl::LogSeverity::kError) {
    fprintf(stderr, "%s\n", message_text.c_str());
    fflush(stderr);
  }
  printf("%s\n", message_text.c_str());
  fflush(stdout);
}

void LogMessage::Flush() {
  std::string message_text = stream_.str();
  SendToLog(message_text);
  stream_.clear();
}

LogMessageFatal::~LogMessageFatal() {
  Flush();
  abort();
}

}  // namespace logging_internal

}  // namespace base
}  // namespace differential_privacy

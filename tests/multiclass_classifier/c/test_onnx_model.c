#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <wchar.h>
#include <locale.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>
#include <pthread.h>
#include <math.h>
#include "onnxruntime-osx-universal2-1.22.0/include/onnxruntime_c_api.h"
#include <cjson/cJSON.h>

#ifdef __APPLE__
#include <sys/sysctl.h>
#include <mach/mach.h>
#include <mach/mach_host.h>
#include <mach/host_info.h>
#endif

const OrtApi* g_ort = NULL;

// Performance monitoring structures
typedef struct {
    double total_time_ms;
    double preprocessing_time_ms;
    double inference_time_ms;
    double postprocessing_time_ms;
    double throughput_per_sec;
} TimingMetrics;

typedef struct {
    double memory_start_mb;
    double memory_end_mb;
    double memory_delta_mb;
    double cpu_avg_percent;
    double cpu_max_percent;
    int cpu_readings_count;
    double* cpu_readings;
} ResourceMetrics;

typedef struct {
    char platform[256];
    char processor[256];
    int cpu_count_physical;
    int cpu_count_logical;
    double total_memory_gb;
    char python_version[64];
} SystemInfo;

typedef struct {
    int monitoring;
    double* cpu_readings;
    int readings_count;
    int max_readings;
    pthread_mutex_t mutex;
} CPUMonitor;

// Global CPU monitor
CPUMonitor g_cpu_monitor = {0};

// Utility functions for performance monitoring
double get_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

double get_memory_usage_mb() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss / 1024.0; // Convert KB to MB on Linux, already MB on macOS
}

double get_cpu_usage_percent() {
#ifdef __APPLE__
    host_cpu_load_info_data_t cpuinfo;
    mach_msg_type_number_t count = HOST_CPU_LOAD_INFO_COUNT;
    if (host_statistics(mach_host_self(), HOST_CPU_LOAD_INFO, (host_info_t)&cpuinfo, &count) == KERN_SUCCESS) {
        unsigned long total_ticks = 0;
        for (int i = 0; i < CPU_STATE_MAX; i++) {
            total_ticks += cpuinfo.cpu_ticks[i];
        }
        return total_ticks > 0 ? ((double)(cpuinfo.cpu_ticks[CPU_STATE_USER] + cpuinfo.cpu_ticks[CPU_STATE_SYSTEM]) / total_ticks) * 100.0 : 0.0;
    }
#endif
    return 0.0; // Fallback
}

void get_system_info(SystemInfo* info) {
    // Platform information
    strcpy(info->platform, "macOS/Linux");
    strcpy(info->processor, "Unknown");
    strcpy(info->python_version, "N/A (C Implementation)");
    
#ifdef __APPLE__
    size_t size = sizeof(info->cpu_count_physical);
    sysctlbyname("hw.physicalcpu", &info->cpu_count_physical, &size, NULL, 0);
    sysctlbyname("hw.logicalcpu", &info->cpu_count_logical, &size, NULL, 0);
    
    uint64_t memsize;
    size = sizeof(memsize);
    sysctlbyname("hw.memsize", &memsize, &size, NULL, 0);
    info->total_memory_gb = memsize / (1024.0 * 1024.0 * 1024.0);
    
    char processor_brand[256];
    size = sizeof(processor_brand);
    if (sysctlbyname("machdep.cpu.brand_string", processor_brand, &size, NULL, 0) == 0) {
        strncpy(info->processor, processor_brand, sizeof(info->processor) - 1);
    }
#else
    info->cpu_count_physical = sysconf(_SC_NPROCESSORS_ONLN);
    info->cpu_count_logical = sysconf(_SC_NPROCESSORS_ONLN);
    
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    info->total_memory_gb = (pages * page_size) / (1024.0 * 1024.0 * 1024.0);
#endif
}

void* cpu_monitor_thread(void* arg) {
    while (g_cpu_monitor.monitoring) {
        double cpu_usage = get_cpu_usage_percent();
        
        pthread_mutex_lock(&g_cpu_monitor.mutex);
        if (g_cpu_monitor.readings_count < g_cpu_monitor.max_readings) {
            g_cpu_monitor.cpu_readings[g_cpu_monitor.readings_count++] = cpu_usage;
        }
        pthread_mutex_unlock(&g_cpu_monitor.mutex);
        
        usleep(10000); // 10ms
    }
    return NULL;
}

void start_cpu_monitoring(int max_readings) {
    g_cpu_monitor.cpu_readings = malloc(max_readings * sizeof(double));
    g_cpu_monitor.readings_count = 0;
    g_cpu_monitor.max_readings = max_readings;
    g_cpu_monitor.monitoring = 1;
    pthread_mutex_init(&g_cpu_monitor.mutex, NULL);
    
    pthread_t thread;
    pthread_create(&thread, NULL, cpu_monitor_thread, NULL);
    pthread_detach(thread);
}

void stop_cpu_monitoring(ResourceMetrics* metrics) {
    g_cpu_monitor.monitoring = 0;
    usleep(50000); // Wait 50ms for thread to finish
    
    pthread_mutex_lock(&g_cpu_monitor.mutex);
    
    if (g_cpu_monitor.readings_count > 0) {
        double sum = 0.0, max_val = 0.0;
        for (int i = 0; i < g_cpu_monitor.readings_count; i++) {
            sum += g_cpu_monitor.cpu_readings[i];
            if (g_cpu_monitor.cpu_readings[i] > max_val) {
                max_val = g_cpu_monitor.cpu_readings[i];
            }
        }
        metrics->cpu_avg_percent = sum / g_cpu_monitor.readings_count;
        metrics->cpu_max_percent = max_val;
        metrics->cpu_readings_count = g_cpu_monitor.readings_count;
        
        // Copy readings for detailed analysis
        metrics->cpu_readings = malloc(g_cpu_monitor.readings_count * sizeof(double));
        memcpy(metrics->cpu_readings, g_cpu_monitor.cpu_readings, g_cpu_monitor.readings_count * sizeof(double));
    } else {
        metrics->cpu_avg_percent = 0.0;
        metrics->cpu_max_percent = 0.0;
        metrics->cpu_readings_count = 0;
        metrics->cpu_readings = NULL;
    }
    
    free(g_cpu_monitor.cpu_readings);
    pthread_mutex_unlock(&g_cpu_monitor.mutex);
    pthread_mutex_destroy(&g_cpu_monitor.mutex);
}

// Function to convert Cyrillic uppercase to lowercase
void cyrillic_to_lower(char* text) {
    unsigned char* p = (unsigned char*)text;
    while (*p) {
        if (*p == 0xD0 && *(p+1)) {
            // Cyrillic А-Я range: 0xD090-0xD0AF -> 0xD0B0-0xD0CF
            if (*(p+1) >= 0x90 && *(p+1) <= 0xAF) {
                *(p+1) += 0x20;  // Convert uppercase to lowercase
            }
            // Handle Р-Я: 0xD080-0xD08F -> 0xD190-0xD19F  
            else if (*(p+1) >= 0x80 && *(p+1) <= 0x8F) {
                *p = 0xD1;       // Change first byte
                *(p+1) += 0x10;  // Adjust second byte
            }
            p += 2;
        } else if (*p == 0xD1 && *(p+1)) {
            // Already lowercase Cyrillic or other D1 range
            p += 2;
        } else if (*p < 0x80) {  // ASCII characters
            *p = tolower(*p);
            p++;
        } else {
            p++;  // Skip other bytes
        }
    }
}

int32_t* preprocess_text(const char* text, const char* tokenizer_file) {
    int32_t* vector = calloc(30, sizeof(int32_t));

    FILE* f = fopen(tokenizer_file, "r");
    if (!f) return NULL;
    
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* json_str = malloc(len + 1);
    fread(json_str, 1, len, f);
    json_str[len] = 0;
    fclose(f);

    cJSON* tokenizer = cJSON_Parse(json_str);
    if (!tokenizer) {
        free(json_str);
        return NULL;
    }

    char* text_copy = strdup(text);
    
    // Convert Cyrillic text to lowercase for proper tokenization
    cyrillic_to_lower(text_copy);
    
    char* word = strtok(text_copy, " \t\n");
    int idx = 0;
    while (word && idx < 30) {
        cJSON* token = cJSON_GetObjectItem(tokenizer, word);
        vector[idx++] = token ? token->valueint : (cJSON_GetObjectItem(tokenizer, "<OOV>") ? cJSON_GetObjectItem(tokenizer, "<OOV>")->valueint : 1);
        word = strtok(NULL, " \t\n");
    }

    free(text_copy);
    free(json_str);
    cJSON_Delete(tokenizer);
    return vector;
}

void print_system_info(SystemInfo* info) {
    printf("\n💻 SYSTEM INFORMATION:\n");
    printf("   Platform: %s\n", info->platform);
    printf("   CPU: %s\n", info->processor);
    printf("   CPU Cores: %d physical, %d logical\n", info->cpu_count_physical, info->cpu_count_logical);
    printf("   Total Memory: %.1f GB\n", info->total_memory_gb);
    printf("   Implementation: C with ONNX Runtime\n");
    printf("\n");
}

void print_performance_summary(TimingMetrics* timing, ResourceMetrics* resources, int text_count) {
    printf("📈 PERFORMANCE SUMMARY:\n");
    printf("   Total Processing Time: %.2fms\n", timing->total_time_ms);
    printf("   ┣━ Preprocessing: %.2fms (%.1f%%)\n", 
           timing->preprocessing_time_ms, 
           (timing->preprocessing_time_ms / timing->total_time_ms) * 100.0);
    printf("   ┣━ Model Inference: %.2fms (%.1f%%)\n", 
           timing->inference_time_ms, 
           (timing->inference_time_ms / timing->total_time_ms) * 100.0);
    printf("   ┗━ Post-processing: %.2fms (%.1f%%)\n", 
           timing->postprocessing_time_ms, 
           (timing->postprocessing_time_ms / timing->total_time_ms) * 100.0);
    printf("   🧠 CPU Usage: %.1f%% avg, %.1f%% peak (%d readings)\n", 
           resources->cpu_avg_percent, resources->cpu_max_percent, resources->cpu_readings_count);
    printf("   💾 Memory: %.1fMB → %.1fMB (Δ%+.1fMB)\n", 
           resources->memory_start_mb, resources->memory_end_mb, resources->memory_delta_mb);
    printf("   🚀 Throughput: %.1f texts/sec\n", timing->throughput_per_sec);
    
    // Performance classification
    const char* performance_class;
    if (timing->total_time_ms < 10) {
        performance_class = "🚀 EXCELLENT";
    } else if (timing->total_time_ms < 50) {
        performance_class = "✅ GOOD";
    } else if (timing->total_time_ms < 100) {
        performance_class = "⚠️ ACCEPTABLE";
    } else {
        performance_class = "❌ NEEDS OPTIMIZATION";
    }
    
    printf("   Performance Rating: %s\n", performance_class);
    printf("\n");
}

int test_single_text(const char* text, const char* model_path, const char* vocab_path, const char* scaler_path) {
    printf("🔄 Processing: %s\n", text);
    
    // Initialize system info
    SystemInfo system_info;
    get_system_info(&system_info);
    print_system_info(&system_info);
    
    // Initialize timing and resource metrics
    TimingMetrics timing = {0};
    ResourceMetrics resources = {0};
    
    double total_start = get_time_ms();
    resources.memory_start_mb = get_memory_usage_mb();
    
    // Start CPU monitoring
    start_cpu_monitoring(1000);
    
    // Initialize ONNX Runtime
    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!g_ort) {
        printf("❌ Failed to initialize ONNX Runtime API\n");
        return 1;
    }
    
    // Preprocessing
    double preprocess_start = get_time_ms();
    int32_t* vector = preprocess_text(text, vocab_path);
    if (!vector) {
        printf("❌ Failed to preprocess text\n");
        return 1;
    }
    timing.preprocessing_time_ms = get_time_ms() - preprocess_start;
    
    // Model setup
    OrtEnv* env;
    OrtStatus* status = g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env);
    if (status) {
        printf("❌ Failed to create ONNX environment\n");
        return 1;
    }

    OrtSessionOptions* session_options;
    status = g_ort->CreateSessionOptions(&session_options);
    if (status) {
        printf("❌ Failed to create session options\n");
        return 1;
    }

    OrtSession* session;
    status = g_ort->CreateSession(env, model_path, session_options, &session);
    if (status) {
        printf("❌ Failed to create ONNX session\n");
        return 1;
    }

    OrtMemoryInfo* memory_info;
    status = g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
    if (status) {
        printf("❌ Failed to create memory info\n");
        return 1;
    }

    int64_t input_shape[] = {1, 30};
    OrtValue* input_tensor;
    status = g_ort->CreateTensorWithDataAsOrtValue(memory_info, vector, 30 * sizeof(int32_t), input_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, &input_tensor);
    if (status) {
        printf("❌ Failed to create input tensor\n");
        return 1;
    }

    // Model inference
    double inference_start = get_time_ms();
    
    const char* input_names[] = {"input"};
    
    // Get output information
    OrtAllocator* allocator = NULL;
    char* output_name = NULL;
    size_t num_output_nodes;
    
    status = g_ort->SessionGetOutputCount(session, &num_output_nodes);
    if (status) {
        printf("❌ Failed to get output count\n");
        return 1;
    }
    
    status = g_ort->GetAllocatorWithDefaultOptions(&allocator);
    if (status) {
        printf("❌ Failed to get allocator\n");
        return 1;
    }
    
    status = g_ort->SessionGetOutputName(session, 0, allocator, &output_name);
    if (status) {
        printf("❌ Failed to get output name\n");
        return 1;
    }
    
    const char* output_names[] = {output_name};
    OrtValue* output_tensor = NULL;
    status = g_ort->Run(session, NULL, input_names, (const OrtValue* const*)&input_tensor, 1, output_names, 1, &output_tensor);
    if (status) {
        printf("❌ Failed to run inference\n");
        return 1;
    }
    
    timing.inference_time_ms = get_time_ms() - inference_start;
    
    // Post-processing
    double postprocess_start = get_time_ms();
    
    float* output_data;
    status = g_ort->GetTensorMutableData(output_tensor, (void**)&output_data);
    if (status) {
        printf("❌ Failed to get output data\n");
        return 1;
    }

    // Load label mapping
    FILE* f = fopen(scaler_path, "r");
    if (!f) {
        printf("❌ Failed to open scaler file\n");
        return 1;
    }

    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* json_str = malloc(len + 1);
    fread(json_str, 1, len, f);
    json_str[len] = 0;
    fclose(f);

    cJSON* label_map = cJSON_Parse(json_str);
    if (!label_map) {
        printf("❌ Failed to parse label mapping\n");
        free(json_str);
        return 1;
    }

    // Find predicted class
    int predicted_idx = 0;
    float max_prob = output_data[0];
    int num_classes = cJSON_GetArraySize(label_map);
    
    // Find predicted class first
    for (int i = 0; i < num_classes; i++) {
        if (output_data[i] > max_prob) {
            max_prob = output_data[i];
            predicted_idx = i;
        }
    }

    char idx_str[16];
    snprintf(idx_str, sizeof(idx_str), "%d", predicted_idx);
    cJSON* predicted_label = cJSON_GetObjectItem(label_map, idx_str);
    
    timing.postprocessing_time_ms = get_time_ms() - postprocess_start;
    
    // Final measurements
    timing.total_time_ms = get_time_ms() - total_start;
    timing.throughput_per_sec = 1000.0 / timing.total_time_ms;
    resources.memory_end_mb = get_memory_usage_mb();
    resources.memory_delta_mb = resources.memory_end_mb - resources.memory_start_mb;
    
    // Stop CPU monitoring
    stop_cpu_monitoring(&resources);
    
    // Display results
    printf("📊 TOPIC CLASSIFICATION RESULTS:\n");
    printf("⏱️  Processing Time: %.1fms\n", timing.total_time_ms);
    
    // Category emojis
    const char* category_emoji(const char* category) {
        if (strcmp(category, "politics") == 0) return "🏛️";
        if (strcmp(category, "technology") == 0) return "💻";
        if (strcmp(category, "sports") == 0) return "⚽";
        if (strcmp(category, "business") == 0) return "💼";
        if (strcmp(category, "entertainment") == 0) return "🎭";
        return "📝";
    }
    
    if (predicted_label) {
        char category_upper[256];
        strcpy(category_upper, predicted_label->valuestring);
        for (int i = 0; category_upper[i]; i++) {
            category_upper[i] = toupper(category_upper[i]);
        }
        
        printf("   🏆 Predicted Category: %s %s\n", category_upper, category_emoji(predicted_label->valuestring));
        printf("   📈 Confidence: %.1f%%\n", max_prob * 100.0);
        printf("   📝 Input Text: \"%s\"\n", text);
        printf("\n");
    }
    
    printf("📊 DETAILED PROBABILITIES:\n");
    for (int i = 0; i < num_classes; i++) {
        char idx_str[16];
        snprintf(idx_str, sizeof(idx_str), "%d", i);
        cJSON* label = cJSON_GetObjectItem(label_map, idx_str);
        if (label) {
            char class_display[256];
            strcpy(class_display, label->valuestring);
            if (class_display[0]) {
                class_display[0] = toupper(class_display[0]);
            }
            
            // Create progress bar
            int bar_length = (int)(output_data[i] * 20);
            char bar[21] = {0};
            for (int j = 0; j < bar_length && j < 20; j++) {
                bar[j] = '█';
            }
            
            const char* star = (i == predicted_idx) ? " ⭐" : "";
            printf("   %s %s: %.1f%% %s%s\n", 
                   category_emoji(label->valuestring),
                   class_display,
                   output_data[i] * 100.0,
                   bar,
                   star);
        }
    }
    printf("\n");
    
    // Print performance summary
    print_performance_summary(&timing, &resources, 1);
    
    // Cleanup
    g_ort->ReleaseValue(input_tensor);
    g_ort->ReleaseValue(output_tensor);
    g_ort->ReleaseMemoryInfo(memory_info);
    g_ort->ReleaseSession(session);
    g_ort->ReleaseSessionOptions(session_options);
    g_ort->ReleaseEnv(env);

    free(vector);
    free(json_str);
    cJSON_Delete(label_map);
    
    if (allocator && output_name) {
        allocator->Free(allocator, output_name);
    }
    
    if (resources.cpu_readings) {
        free(resources.cpu_readings);
    }

    return 0;
}

int run_performance_benchmark(const char* model_path, const char* vocab_path, const char* scaler_path, int num_runs) {
    printf("\n🚀 PERFORMANCE BENCHMARKING (%d runs)\n", num_runs);
    printf("============================================================\n");
    
    SystemInfo system_info;
    get_system_info(&system_info);
    printf("💻 System: %d cores, %.1fGB RAM\n", system_info.cpu_count_physical, system_info.total_memory_gb);
    
    const char* test_text = "This is a sample text for performance testing";
    printf("📝 Test Text: '%s'\n\n", test_text);
    
    // Initialize ONNX Runtime once
    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!g_ort) return 1;
    
    OrtEnv* env;
    OrtStatus* status = g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "benchmark", &env);
    if (status) return 1;

    OrtSessionOptions* session_options;
    status = g_ort->CreateSessionOptions(&session_options);
    if (status) return 1;

    OrtSession* session;
    status = g_ort->CreateSession(env, model_path, session_options, &session);
    if (status) return 1;

    OrtMemoryInfo* memory_info;
    status = g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
    if (status) return 1;
    
    // Preprocess once
    int32_t* vector = preprocess_text(test_text, vocab_path);
    if (!vector) return 1;
    
    int64_t input_shape[] = {1, 30};
    OrtValue* input_tensor;
    status = g_ort->CreateTensorWithDataAsOrtValue(memory_info, vector, 30 * sizeof(int32_t), input_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, &input_tensor);
    if (status) return 1;
    
    const char* input_names[] = {"input"};
    
    // Get output name
    OrtAllocator* allocator = NULL;
    char* output_name = NULL;
    status = g_ort->GetAllocatorWithDefaultOptions(&allocator);
    if (status) return 1;
    status = g_ort->SessionGetOutputName(session, 0, allocator, &output_name);
    if (status) return 1;
    const char* output_names[] = {output_name};
    
    // Warmup runs
    printf("🔥 Warming up model (5 runs)...\n");
    for (int i = 0; i < 5; i++) {
        OrtValue* output_tensor = NULL;
        g_ort->Run(session, NULL, input_names, (const OrtValue* const*)&input_tensor, 1, output_names, 1, &output_tensor);
        g_ort->ReleaseValue(output_tensor);
    }
    
    // Performance arrays
    double* times = malloc(num_runs * sizeof(double));
    double* inference_times = malloc(num_runs * sizeof(double));
    double* memory_usage = malloc(num_runs * sizeof(double));
    
    printf("📊 Running %d performance tests...\n", num_runs);
    double overall_start = get_time_ms();
    
    for (int i = 0; i < num_runs; i++) {
        if (i % 20 == 0 && i > 0) {
            printf("   Progress: %d/%d (%.1f%%)\n", i, num_runs, (double)i / num_runs * 100.0);
        }
        
        double start_memory = get_memory_usage_mb();
        double start_time = get_time_ms();
        
        double inference_start = get_time_ms();
        OrtValue* output_tensor = NULL;
        g_ort->Run(session, NULL, input_names, (const OrtValue* const*)&input_tensor, 1, output_names, 1, &output_tensor);
        double inference_time = get_time_ms() - inference_start;
        
        double end_time = get_time_ms();
        double end_memory = get_memory_usage_mb();
        
        times[i] = end_time - start_time;
        inference_times[i] = inference_time;
        memory_usage[i] = end_memory - start_memory;
        
        g_ort->ReleaseValue(output_tensor);
    }
    
    double overall_time = get_time_ms() - overall_start;
    
    // Calculate statistics
    double sum = 0, inf_sum = 0, mem_sum = 0;
    double min_time = times[0], max_time = times[0];
    double min_inf = inference_times[0], max_inf = inference_times[0];
    
    for (int i = 0; i < num_runs; i++) {
        sum += times[i];
        inf_sum += inference_times[i];
        mem_sum += memory_usage[i];
        
        if (times[i] < min_time) min_time = times[i];
        if (times[i] > max_time) max_time = times[i];
        if (inference_times[i] < min_inf) min_inf = inference_times[i];
        if (inference_times[i] > max_inf) max_inf = inference_times[i];
    }
    
    double avg_time = sum / num_runs;
    double avg_inf = inf_sum / num_runs;
    double avg_mem = mem_sum / num_runs;
    
    // Calculate standard deviation
    double variance = 0;
    for (int i = 0; i < num_runs; i++) {
        variance += (times[i] - avg_time) * (times[i] - avg_time);
    }
    double std_dev = sqrt(variance / num_runs);
    
    // Display results
    printf("\n📈 DETAILED PERFORMANCE RESULTS:\n");
    printf("--------------------------------------------------\n");
    printf("⏱️  TIMING ANALYSIS:\n");
    printf("   Total Time per Text:\n");
    printf("     Mean: %.2fms\n", avg_time);
    printf("     Min: %.2fms\n", min_time);
    printf("     Max: %.2fms\n", max_time);
    printf("     Standard deviation: %.2fms\n", std_dev);
    printf("\n   Model Inference Only:\n");
    printf("     Mean: %.2fms\n", avg_inf);
    printf("     Min: %.2fms\n", min_inf);
    printf("     Max: %.2fms\n", max_inf);
    printf("\n💾 MEMORY USAGE:\n");
    printf("   Average delta: %.2fMB\n", avg_mem);
    printf("   Current usage: %.1fMB\n", get_memory_usage_mb());
    printf("\n🚀 THROUGHPUT:\n");
    printf("   Texts per second: %.1f\n", 1000.0 / avg_time);
    printf("   Total benchmark time: %.2fs\n", overall_time / 1000.0);
    printf("   Overall throughput: %.1f texts/sec\n", num_runs / (overall_time / 1000.0));
    
    // Performance classification
    const char* performance_class;
    if (avg_time < 10) {
        performance_class = "🚀 EXCELLENT";
    } else if (avg_time < 50) {
        performance_class = "✅ GOOD";
    } else if (avg_time < 100) {
        performance_class = "⚠️ ACCEPTABLE";
    } else {
        performance_class = "❌ POOR";
    }
    
    printf("\n🎯 PERFORMANCE CLASSIFICATION: %s\n", performance_class);
    printf("   (%.1fms average - Target: <100ms)\n", avg_time);
    
    // Cleanup
    free(times);
    free(inference_times);
    free(memory_usage);
    
    g_ort->ReleaseValue(input_tensor);
    g_ort->ReleaseMemoryInfo(memory_info);
    g_ort->ReleaseSession(session);
    g_ort->ReleaseSessionOptions(session_options);
    g_ort->ReleaseEnv(env);
    
    free(vector);
    if (allocator && output_name) {
        allocator->Free(allocator, output_name);
    }
    
    return 0;
}

int main(int argc, char* argv[]) {
    printf("🤖 ONNX MULTICLASS CLASSIFIER - C IMPLEMENTATION\n");
    printf("==================================================\n");
    
    const char* model_path = "model.onnx";
    const char* vocab_path = "vocab.json";
    const char* scaler_path = "scaler.json";
    
    if (argc > 1) {
        if (strcmp(argv[1], "--benchmark") == 0) {
            int num_runs = argc > 2 ? atoi(argv[2]) : 100;
            return run_performance_benchmark(model_path, vocab_path, scaler_path, num_runs);
        } else {
            // Use command line argument as text
            return test_single_text(argv[1], model_path, vocab_path, scaler_path);
        }
    } else {
        // Default test
        const char* default_texts[] = {
            "шляк би тебе трафив",
            "This is a health related topic about medicine",
            "The football team won the championship game",
            "Political news about the election results"
        };
        
        printf("🔄 Testing multiple texts...\n");
        for (int i = 0; i < 4; i++) {
            printf("\n--- Test %d/4 ---\n", i + 1);
            test_single_text(default_texts[i], model_path, vocab_path, scaler_path);
        }
        
        // Run benchmark
        printf("\n🚀 Running performance benchmark...\n");
        run_performance_benchmark(model_path, vocab_path, scaler_path, 50);
    }
    
    return 0;
} 
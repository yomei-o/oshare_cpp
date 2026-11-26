// ブラウザ用の入口。重み・prompts を受け取り、RGBA 画素からおしゃれ度とタグを返す。
#include "oshare.hpp"
#include <emscripten/emscripten.h>
#include <string>

static oshare::Weights g_w;
static oshare::Prompts g_p;
static oshare::Result g_r;
static std::string g_json;

extern "C" {
EMSCRIPTEN_KEEPALIVE int os_load_weights(const uint8_t* p, int n) {
    try { g_w = oshare::Weights(); g_w.load(p, (size_t)n); return 1; } catch (...) { return 0; }
}
EMSCRIPTEN_KEEPALIVE int os_load_prompts(const char* p, int n) {
    try { g_p = oshare::Prompts(); g_p.load(std::string(p, (size_t)n)); return 1; } catch (...) { return 0; }
}
// rgba: w*h*4。戻り値は結果の JSON 文字列（UTF-8、次の呼び出しまで有効）
EMSCRIPTEN_KEEPALIVE const char* os_judge(const uint8_t* rgba, int w, int h) {
    oshare::Image im; im.w = w; im.h = h; im.px.resize((size_t)w * h * 3);
    for (size_t i = 0; i < (size_t)w * h; i++) for (int c = 0; c < 3; c++) im.px[i * 3 + c] = rgba[i * 4 + c];
    g_r = oshare::judge(g_p, oshare::encode_image(g_w, oshare::preprocess(im, g_p.pad)));
    char buf[64];
    g_json = "{\"score\":"; std::snprintf(buf, sizeof buf, "%.3f", g_r.score); g_json += buf;
    g_json += ",\"raw\":"; std::snprintf(buf, sizeof buf, "%.5f", g_r.raw); g_json += buf;
    g_json += ",\"tags\":[";
    for (size_t i = 0; i < g_r.tags.size(); i++) {
        std::snprintf(buf, sizeof buf, "%.5f", g_r.tags[i].second);
        g_json += (i ? ",[\"" : "[\"") + g_r.tags[i].first + "\"," + buf + "]";
    }
    g_json += "]}";
    return g_json.c_str();
}
}

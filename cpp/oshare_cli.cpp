// oshare_cli — 写真のおしゃれ度とスタイルタグを出す（python/oshare.py の CLI と同じ出力）。
// usage: oshare_cli weights.bin prompts.json image... [--feat]
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "oshare.hpp"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <chrono>

static std::string slurp(const char* fn) {
    std::ifstream f(fn, std::ios::binary);
    if (!f) throw std::runtime_error(std::string("cannot open ") + fn);
    std::ostringstream ss; ss << f.rdbuf(); return ss.str();
}

int main(int argc, char** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: oshare_cli weights.bin prompts.json image... [--feat]\n"); return 1; }
    try {
        bool feat = false;
        for (int i = 3; i < argc; i++) if (std::string(argv[i]) == "--feat") feat = true;
        std::string wb = slurp(argv[1]);
        oshare::Weights w; w.load((const uint8_t*)wb.data(), wb.size()); wb.clear(); wb.shrink_to_fit();
        oshare::Prompts P; P.load(slurp(argv[2]));
        for (int i = 3; i < argc; i++) {
            if (argv[i][0] == '-' && argv[i][1] == '-') continue;
            int W, H, C;
            unsigned char* d = stbi_load(argv[i], &W, &H, &C, 3);
            if (!d) { std::fprintf(stderr, "cannot load %s\n", argv[i]); continue; }
            oshare::Image im; im.w = W; im.h = H; im.px.assign(d, d + (size_t)W * H * 3); stbi_image_free(d);
            auto t0 = std::chrono::steady_clock::now();
            auto f = oshare::encode_image(w, oshare::preprocess(im, P.pad));
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (feat) { for (size_t k = 0; k < f.size(); k++) std::printf(k ? " %.6f" : "%.6f", f[k]); std::printf("\n"); continue; }
            auto r = oshare::judge(P, f);
            std::printf("%s\tscore=%.1f raw=%.4f\t", argv[i], r.score, r.raw);
            for (int k = 0; k < 3 && k < (int)r.tags.size(); k++) std::printf("%s%s:%.2f", k ? " " : "", r.tags[k].first.c_str(), r.tags[k].second);
            std::printf("\t(%.0f ms)\n", ms);
        }
    } catch (std::exception& e) { std::fprintf(stderr, "error: %s\n", e.what()); return 1; }
    return 0;
}

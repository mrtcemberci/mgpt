#include <iostream>
#include "pipeline/CLIConfig.h"
#include "pipeline/Trainer.h"
#include "pipeline/Generator.h"

int main(int argc, char** argv) {
    try {
        std::cout << "============================================================\n";
        std::cout << "      MGPT BOOTING UP...      \n";
        std::cout << "============================================================\n\n";

        CLIConfig config = parse_arguments(argc, argv);
        if (config.show_help) {
            return 0;
        }

        if (config.mode_infer_only) {
            Generator generator(config);
            return generator.run();
        } else {
            Trainer trainer(config);
            return trainer.run();
        }
    } catch (const std::exception& e) {
        std::cerr << "\n[FATAL ERROR]: " << e.what() << "\n";
        return -1;
    }
}

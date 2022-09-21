# CMakeList.txt: projekt CMake dla elementu agds_ref, dołącz źródło i zdefiniuj
# tutaj logikę specyficzną dla projektu.
#
cmake_minimum_required (VERSION 3.8)

# Dodaj źródło do pliku wykonywalnego tego projektu.
add_executable (agds_ref "agds_ref.cpp" "agds_ref.h" "mock_agds_data.hpp" "mock_agds_data.cpp" "agds_components.hpp"  "Agds.cpp" "Conn.cpp" "utils.hpp" "utils.cpp" "measurements.hpp" "measurements.cpp" "inference_old.hpp" "inference_old.cpp" "ProductVec.cpp"  "avbtree.hpp" "agds.hpp" "inference.cpp" "inference.hpp")

# TODO: Dodaj testy i zainstaluj elementy docelowe w razie potrzeby.

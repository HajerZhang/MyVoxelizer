file(GLOB_RECURSE TINYXML_LIST CONFIGURE_DEPENDS *.cpp *.h)

add_library(tinyxml2 SHARED ${TINYXML_LIST})

target_include_directories(
    tinyxml2
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
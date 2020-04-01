set(FILES "${CMAKE_CURRENT_SOURCE_DIR}/tf_tensor.h", "${CMAKE_CURRENT_SOURCE_DIR}/tf_status.h","${CMAKE_CURRENT_SOURCE_DIR}/tf_datatype.h", "${CMAKE_CURRENT_SOURCE_DIR}/tf_attrtype.h", "${CMAKE_CURRENT_SOURCE_DIR}/c_api_experimental.h", "${CMAKE_CURRENT_SOURCE_DIR}/c_api.h")
set(SUBDIRECTORIES "eager")
set(RELATIVE_PATH "include/tensorflow/c")

#Add all subdirectories 
foreach(VAR ${SUBDIRECTORIES})
    add_subdirectory(${VAR})
endforeach()

if(FILES)
    #Add source files to target
    target_sources("${PROJECT_NAME}" PRIVATE ${FILES})
    
    #Install our source files
    install(FILES ${FILES} DESTINATION ${RELATIVE_PATH})
endif()

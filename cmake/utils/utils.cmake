macro(__knowhere_option variable description value)
  if(NOT DEFINED ${variable})
    set(${variable}
        ${value}
        CACHE STRING ${description})
  endif()
endmacro()

set(KNOWHERE_ALL_OPTIONS)

macro(knowhere_option variable description value)
  set(__value ${value})
  set(__condition "")
  set(__varname "__value")
  list(APPEND knowhere_ALL_OPTIONS ${variable})

  foreach(arg ${ARGN})
    if(arg STREQUAL "IF" OR arg STREQUAL "if")
      set(__varname "__condition")
    else()
      list(APPEND ${__varname} ${arg})
    endif()
  endforeach()

  unset(__varname)

  if("${__condition}" STREQUAL "")
    set(__condition 2 GREATER 1)
  endif()

  if(${__condition})
    if("${__value}" MATCHES ";")
      if(${__value})
        __knowhere_option(${variable} "${description}" ON)
      else()
        __knowhere_option(${variable} "${description}" OFF)
      endif()
    elseif(DEFINED ${__value})
      if(${__value})
        __knowhere_option(${variable} "${description}" ON)
      else()
        __knowhere_option(${variable} "${description}" OFF)
      endif()
    else()
      __knowhere_option(${variable} "${description}" "${__value}")
    endif()
  else()
    unset(${variable} CACHE)
  endif()
endmacro()

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.12.0")
  macro(knowhere_file_glob glob variable)
    file(${glob} ${variable} CONFIGURE_DEPENDS ${ARGN})
  endmacro()
else()
  macro(knowhere_file_glob)
    file(${glob} ${variable} ${ARGN})
  endmacro()
endif()

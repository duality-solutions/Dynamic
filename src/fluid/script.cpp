// Copyright (c) 2021 Duality Blockchain Solutions

#include "fluid/script.h"

#include "utilstrencodings.h"

bool WithinFluidRange(opcodetype op)
{
    return op >= OP_MINT && op <= OP_RESERVED_0006;
}

std::string TranslationTable(opcodetype op)
{
    return GetOpName(op);
}

uint64_t TranslationTable(std::string str)
{
    str = SanitizeString(str, SAFE_CHARS_SCRIPT); // Remove invalid characters
         if (str == "OP_MINT")                    { return OP_MINT; }
    else if (str == "OP_REWARD_DYNODE")           { return OP_REWARD_DYNODE; }
    else if (str == "OP_REWARD_MINING")           { return OP_REWARD_MINING; }
    else if (str == "OP_SWAP_SOVEREIGN_ADDRESS")  { return OP_SWAP_SOVEREIGN_ADDRESS; }
    else if (str == "OP_UPDATE_FEES")             { return OP_UPDATE_FEES; }
    else if (str == "OP_FREEZE_ADDRESS")          { return OP_FREEZE_ADDRESS; }
    else if (str == "OP_RELEASE_ADDRESS")         { return OP_RELEASE_ADDRESS; }
    else if (str == "OP_BDAP_REVOKE")             { return OP_BDAP_REVOKE; }
    else if (str == "OP_RESERVED_0001")           { return OP_RESERVED_0001; }
    else if (str == "OP_RESERVED_0002")           { return OP_RESERVED_0002; }
    else if (str == "OP_RESERVED_0003")           { return OP_RESERVED_0003; }
    else if (str == "OP_RESERVED_0004")           { return OP_RESERVED_0004; }
    else if (str == "OP_RESERVED_0005")           { return OP_RESERVED_0005; }
    else if (str == "OP_RESERVED_0006")           { return OP_RESERVED_0006; }
    else                                          { return OP_INVALIDOPCODE; }
}

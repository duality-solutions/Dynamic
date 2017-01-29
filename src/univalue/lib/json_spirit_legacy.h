#ifndef JSON_SPIRIT_LEGACY_H
#define JSON_SPIRIT_LEGACY_H


/* -----------------------------------------------------------------------
 * -------------------------------- write --------------------------------
 * -----------------------------------------------------------------------
 */

template < class String_type >
String_type to_str( const char* c_str )
{
    String_type result;

    for( const char* p = c_str; *p != 0; ++p )
    {
        result += *p;
    }

    return result;
}

inline char to_hex_char( unsigned int c )
{
    assert( c <= 0xF );

    const char ch = static_cast< char >( c );

    if( ch < 10 ) return '0' + ch;

    return 'A' - 10 + ch;
}

template< class String_type >
String_type non_printable_to_string( unsigned int c )
{
    // Silence the warning: typedef ‘Char_type’ locally defined but not used [-Wunused-local-typedefs]
    // typedef typename String_type::value_type Char_type;

    String_type result( 6, '\\' );

    result[1] = 'u';

    result[ 5 ] = to_hex_char( c & 0x000F ); c >>= 4;
    result[ 4 ] = to_hex_char( c & 0x000F ); c >>= 4;
    result[ 3 ] = to_hex_char( c & 0x000F ); c >>= 4;
    result[ 2 ] = to_hex_char( c & 0x000F );

    return result;
}

template< typename Char_type, class String_type >
bool add_esc_char( Char_type c, String_type& s )
{
    switch( c )
    {
        case '"':  s += to_str< String_type >( "\\\"" ); return true;
        case '\\': s += to_str< String_type >( "\\\\" ); return true;
        case '\b': s += to_str< String_type >( "\\b"  ); return true;
        case '\f': s += to_str< String_type >( "\\f"  ); return true;
        case '\n': s += to_str< String_type >( "\\n"  ); return true;
        case '\r': s += to_str< String_type >( "\\r"  ); return true;
        case '\t': s += to_str< String_type >( "\\t"  ); return true;
    }

    return false;
}

template< class String_type >
String_type add_esc_chars( const String_type& s )
{
    typedef typename String_type::const_iterator Iter_type;
    typedef typename String_type::value_type     Char_type;

    String_type result;

    const Iter_type end( s.end() );

    for( Iter_type i = s.begin(); i != end; ++i )
    {
        const Char_type c( *i );

        if( add_esc_char( c, result ) ) continue;

        const wint_t unsigned_c( ( c >= 0 ) ? c : 256 + c );

        if( iswprint( unsigned_c ) )
        {
            result += c;
        }
        else
        {
            result += non_printable_to_string< String_type >( unsigned_c );
        }
    }

    return result;
}

/* ----------------------------------------------------------------------
 * -------------------------------- read --------------------------------
 * ----------------------------------------------------------------------
 */

template< class Char_type >
Char_type hex_to_num( const Char_type c )
{
    if( ( c >= '0' ) && ( c <= '9' ) ) return c - '0';
    if( ( c >= 'a' ) && ( c <= 'f' ) ) return c - 'a' + 10;
    if( ( c >= 'A' ) && ( c <= 'F' ) ) return c - 'A' + 10;
    return 0;
}

template< class Char_type, class Iter_type >
Char_type hex_str_to_char( Iter_type& begin )
{
    const Char_type c1( *( ++begin ) );
    const Char_type c2( *( ++begin ) );

    return ( hex_to_num( c1 ) << 4 ) + hex_to_num( c2 );
}

template< class Char_type, class Iter_type >
Char_type unicode_str_to_char( Iter_type& begin )
{
    const Char_type c1( *( ++begin ) );
    const Char_type c2( *( ++begin ) );
    const Char_type c3( *( ++begin ) );
    const Char_type c4( *( ++begin ) );

    return ( hex_to_num( c1 ) << 12 ) +
           ( hex_to_num( c2 ) <<  8 ) +
           ( hex_to_num( c3 ) <<  4 ) +
           hex_to_num( c4 );
}

template< class String_type >
void append_esc_char_and_incr_iter( String_type& s,
                                    typename String_type::const_iterator& begin,
                                    typename String_type::const_iterator end )
{
    typedef typename String_type::value_type Char_type;

    const Char_type c2( *begin );

    switch( c2 )
    {
        case 't':  s += '\t'; break;
        case 'b':  s += '\b'; break;
        case 'f':  s += '\f'; break;
        case 'n':  s += '\n'; break;
        case 'r':  s += '\r'; break;
        case '\\': s += '\\'; break;
        case '/':  s += '/';  break;
        case '"':  s += '"';  break;
        case 'x':
        {
            if( end - begin >= 3 )  //  expecting "xHH..."
            {
                s += hex_str_to_char< Char_type >( begin );
            }
            break;
        }
        case 'u':
        {
            if( end - begin >= 5 )  //  expecting "uHHHH..."
            {
                s += unicode_str_to_char< Char_type >( begin );
            }
            break;
        }
    }
}

template< class String_type >
String_type substitute_esc_chars( typename String_type::const_iterator begin,
                               typename String_type::const_iterator end )
{
    typedef typename String_type::const_iterator Iter_type;

    if( end - begin < 2 ) return String_type( begin, end );

    String_type result;

    result.reserve( end - begin );

    const Iter_type end_minus_1( end - 1 );

    Iter_type substr_start = begin;
    Iter_type i = begin;

    for( ; i < end_minus_1; ++i )
    {
        if( *i == '\\' )
        {
            result.append( substr_start, i );

            ++i;  // skip the '\'

            append_esc_char_and_incr_iter( result, i, end );

            substr_start = i + 1;
        }
    }

    result.append( substr_start, end );

    return result;
}

#endif // JSON_SPIRIT_LEGACY_H
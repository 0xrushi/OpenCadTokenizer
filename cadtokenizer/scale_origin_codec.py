"""Self-contained codec-local token packets; SEM remains the CAD I/O format.

Decoder callbacks receive code IDs and pair position only, never source SEM.
Kind 0: learned codes. Kind 1: learned codes + three six-bit literals per pair.
Kind 2: exact existing SEM symbols, for the lossless length/fidelity control.
"""
import math
import struct
import zlib

from .representation import split_pairs,stringify,tokenize

HEADER=struct.Struct('>4sBHI')
MAGIC=b'CTP1'
VQ,HYBRID,LOSSLESS=0,1,2


def vocabulary(kind,codebook_size):
    if kind not in (VQ,HYBRID,LOSSLESS) or not 1<=codebook_size<=65535:
        raise ValueError('Invalid packet kind/codebook size')
    if kind==LOSSLESS and codebook_size!=76:raise ValueError('SEM control has 76 non-padding symbols')
    return codebook_size+(64 if kind==HYBRID else 0)+3


def pack(kind,codebook_size,pairs,metadata=None):
    count=vocabulary(kind,codebook_size);end,bos,eos=count-3,count-2,count-1
    if not pairs:raise ValueError('No pairs')
    if kind==HYBRID and (metadata is None or len(metadata)!=len(pairs)):raise ValueError('Missing per-pair metadata')
    if kind!=HYBRID and metadata is not None:raise ValueError('Unexpected metadata')
    tokens=[bos]
    for i,codes in enumerate(pairs):
        if not codes or any(type(v) is not int or not 0<=v<codebook_size for v in codes):raise ValueError('Invalid payload IDs')
        tokens.extend(codes)
        if kind==HYBRID:
            values=metadata[i]
            if len(values)!=3 or any(type(v) is not int or not 0<=v<64 for v in values):raise ValueError('Invalid scale/origin bins')
            tokens.extend(codebook_size+v for v in values)
        tokens.append(end)
    tokens.append(eos)
    width=(count-1).bit_length();acc=bits=0;body=bytearray()
    for token in tokens:
        acc=(acc<<width)|token;bits+=width
        while bits>=8:
            bits-=8;body.append((acc>>bits)&255)
        acc&=(1<<bits)-1
    if bits:body.append(acc<<(8-bits))
    packet=HEADER.pack(MAGIC,kind,codebook_size,len(tokens))+body
    return packet+struct.pack('>I',zlib.crc32(packet))


def unpack(packet):
    if len(packet)<HEADER.size+4:raise ValueError('Truncated packet')
    if zlib.crc32(packet[:-4])!=struct.unpack('>I',packet[-4:])[0]:raise ValueError('Packet CRC mismatch')
    magic,kind,size,n=HEADER.unpack(packet[:HEADER.size])
    if magic!=MAGIC:raise ValueError('Unknown packet version')
    count=vocabulary(kind,size);width=(count-1).bit_length()
    if n<3 or n>1_000_000 or len(packet)!=HEADER.size+(n*width+7)//8+4:raise ValueError('Invalid packet length')
    raw=packet[HEADER.size:-4];integer=int.from_bytes(raw,'big');padding=len(raw)*8-n*width
    if padding and integer&((1<<padding)-1):raise ValueError('Noncanonical bit padding')
    integer>>=padding;mask=(1<<width)-1
    tokens=[(integer>>((n-i-1)*width))&mask for i in range(n)]
    if any(v>=count for v in tokens):raise ValueError('Out-of-vocabulary token')
    end,bos,eos=count-3,count-2,count-1
    if tokens[0]!=bos or tokens[-1]!=eos:raise ValueError('Missing stream boundaries')
    chunks=[];current=[]
    for token in tokens[1:-1]:
        if token==end:
            if not current:raise ValueError('Empty pair')
            chunks.append(current);current=[]
        else:current.append(token)
    if current or not chunks:raise ValueError('Missing pair boundary')
    pairs=[];metadata=[]
    for chunk in chunks:
        if kind==HYBRID:
            if len(chunk)<4 or any(not size<=v<size+64 for v in chunk[-3:]):raise ValueError('Missing metadata trailer')
            metadata.append([v-size for v in chunk[-3:]]);chunk=chunk[:-3]
        if any(not 0<=v<size for v in chunk):raise ValueError('Non-code token in code payload')
        pairs.append(chunk)
    return {'kind':kind,'codebook_size':size,'pairs':pairs,'metadata':metadata if kind==HYBRID else None,
            'tokens':tokens,'token_count':n,'vocabulary_size':count,'bits_per_token':width,
            'payload_bits':n*width,'serialized_bytes':len(packet),'framing_tokens':2+len(pairs),
            'metadata_tokens':3*len(pairs) if kind==HYBRID else 0,'fixed_header_crc_bytes':HEADER.size+4}


def metadata_for_pair(pair):
    boundary=next(i for i,t in enumerate(pair) if t in (8,9,10))
    if len(pair)-boundary!=19:raise ValueError('Invalid extrusion arity')
    values=[v-13 for v in pair[boundary+15:boundary+18]]
    if len(values)!=3 or any(not 0<=v<64 for v in values):raise ValueError('Invalid source metadata')
    return values


def merge_pair(pair,metadata):
    pair=list(pair);parts=split_pairs(pair)
    if len(parts)!=1:raise ValueError('Decoder must return one complete pair')
    boundary=next(i for i,t in enumerate(pair) if t in (8,9,10))
    if len(pair)-boundary!=19:raise ValueError('Decoded extrusion has wrong arity')
    pair[boundary+15:boundary+18]=[v+13 for v in metadata]
    return pair


def decode(packet,pair_decoder=None,expected_codebook_size=None):
    parsed=unpack(packet)
    if expected_codebook_size is not None and parsed['codebook_size']!=expected_codebook_size:
        raise ValueError('Decoder/codebook mismatch')
    if parsed['kind']==LOSSLESS:
        return stringify([v+1 for pair in parsed['pairs'] for v in pair])
    if pair_decoder is None:raise ValueError('Learned packet needs a model decoder')
    result=[]
    for i,codes in enumerate(parsed['pairs']):
        pair=pair_decoder(codes,i==0)
        if parsed['kind']==HYBRID:pair=merge_pair(pair,parsed['metadata'][i])
        result.extend(pair)
    split_pairs(result)
    return stringify(result)


def lossless_packet(sequence):
    return pack(LOSSLESS,76,[[v-1 for v in pair] for pair in split_pairs(tokenize(sequence))])

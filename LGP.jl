__precompile__()

module LGP

using MATLAB
using PyCall

import Base: find, getindex

export
    # methods
    readPatInfo,
    find,
    getindex,
    get_shots,
    affmap,
    scans,
    readlgpfiles,

    # types
    Volume,
    AffineMap,
    Patient

    const zipfile = PyNULL()

    __init__() = copy!(zipfile, pyimport("zipfile"))


"""
    readPatInfo(patient_directory::AbstractString) -> AbstractString

Read and return content of the `PatInfo.xml` file located in the `patient_directory`.
Note output is stripped of `segmented_skulls` and `snapshots` objects, as those ones are gigantic and not needed.
"""
function readPatInfo(patient_directory::AbstractString)

    # ensure that patient_directory ends with '/'
    (patient_directory[end] == '/') || (patient_directory = patient_directory * "/")
    content = open(readstring, patient_directory * "PatInfo.xml")
    # segmented_skulls and snapshots objects together take up +90% of the file and are usless, so remove them
    # remove segmented_skulls object
    content = replace(content, r"<segmented_skulls([\s\S]*?)>segmented_skulls", "")
    # remove snapshots objects
    content = replace(content, r"<snapshots([\s\S]*?)>snapshots", "")

    return content::AbstractString
end


"""
    find_children(parent::AbstractString, child::AbstractString) -> Vector{<:AbstractString}

Find all instances of node `child` in object or string `parent`.
"""
find_children(parent::AbstractString, child::AbstractString) = matchall( Regex("<$(child)([\\s\\S]*?)>$(child)"), parent )::Vector{<:AbstractString}

"""
    finder(child_0::S₁, generations::Vector{S₂}, dummy::BitArray{N}) where {S₁ <: AbstractString, S₂ <: AbstractString, N} -> Vector{String}

Retrieve Nth progeny, i.e. an "Nth generation" of subobjects of object `child_0`,
by generating and evaluating corresponding expression with N nested for loops,
where `N` is determined by the dimensionality of `dummy` and is equal to the length of the `generations`,
output's hierarchy in the object `child_0`.
"""
@generated function finder(child_0::S₁, generations::Vector{S₂}, dummy::BitArray{N}) where {S₁ <: AbstractString, S₂ <: AbstractString, N}

    # generate symbols of iteration variables in nested for loops
    itervars = [ Symbol(:child_,dim) for dim ∈ 0:(N-1) ]

    # generate body of the innermost for loop
    expr = quote
        append!( progeny, find_children($(itervars[N]), generations[$N]) )
    end

    # generate all nested for loops
    for dim ∈ (N-1):-1:1
        expr = quote
            for $(itervars[dim+1]) ∈ find_children($(itervars[dim]), generations[$(dim)])
                $expr
            end
        end
    end

    # add pre and post for loops expressions
    expr = quote
        progeny = Array{String}(0)
        $expr
        progeny
    end
    expr
end

"""
    find(object::AbstractString, node::AbstractString) -> Vector{String}

Find all instances of the `node` in the `object`.
`node` can be a name of a single node or a hierarchy of node names delimited by `/`.
"""
function Base.find(object::AbstractString, node::AbstractString)

    # get the nodes "ancestry"
    nodechain = split(node, "/")
    # ensure that no empty object names are passed
    contains(==,isempty.(nodechain), true) && error("one or more object names in node chain are empty")

    # generate dummy for calling finder
    eval( parse( "dummy = falses(" * "0,"^length(nodechain) * ")" ) )

    # get the all instances of the node in the object
    nodes = finder(object, nodechain, dummy)

    return nodes::Vector{String}
end


"""
    getindex(object::AbstractString, attribute::AbstractString, prefix::Char='#') -> AbstractString

Retrieve the value(s) stored in PatInfo.xml file for the given `attribute` within an `object`.
"""
function Base.getindex(object::AbstractString, attribute::AbstractString, prefix::Char='#')
    esc = (prefix == '*' || prefix == '+') ? "\\" : ""
    m = match( Regex(esc * "$(prefix)$(attribute)=(\\d*):(.*)\n"), object )
    if typeof(m) == Void
        object_name = match(r"<(.*?)\n", object).captures[1]
        error("object `$(object_name)` has no filed with the name `$(prefix)$(attribute)`")
    end
    return m.captures[end]::AbstractString
end


"""
    get_shots(target::String) -> Matrix{Float64}

Get set of shot positions, i.e. isocenters that were planned for the `taret` node from the PatInfo.xml file.
"""
function get_shots(target::String)
    shots = find(target, "shots")
    # initialize
    isocenters = zeros(3, length(shots))
    # populate
    for (i, shot) ∈ enumerate(shots)
        isocenters[:,i] = float( [ shot["x"]; shot["y"]; shot["z"] ] )
    end
    return isocenters::Matrix{Float64}
end

"""
    scans(patient_directory::AbstractString) -> Vector{Array{Int,3}}

Construct from DICOM image files, as vector of image tensors whose elements are 3D brain scan (such as MRI, CT, etc.) tensors.
`patient_directory` must be a path to the folder extracted from `.lgp` patient file that contains PatInfo.xml file and DICOM image files.
"""
function scans(patient_directory::AbstractString)

    # ensure that patient_directory ends with '/'
    (patient_directory[end] == '/') || (patient_directory = patient_directory * "/")

    # read PatInfo.xml file
    content = readPatInfo(patient_directory)

    ### construct 3D image tensor for each stack of images
    image_stacks = find(content, "patients/examinations/image_stacks")
    image_tensors = Array{Array{Int64,3}}(length(image_stacks))

    for (n, image_stack) ∈ enumerate(image_stacks)
        images = find(image_stack, "images")

        image_tensors[n] = let
            filepath = patient_directory * images[1]["sop_instance_uid"]
            s = size(mat"dicomread($filepath)")
            zeros(Int, s[1], s[2], length(images))
        end

        # read DICOM image slices using MATLAB's dicomread() function
        for (k, image) ∈ enumerate(images)
            filepath = patient_directory * image["sop_instance_uid"]
            image_tensors[n][:,:,k] = convert( Array{Int}, mat"dicomread($filepath)" )
        end
    end

    return image_tensors::Vector{Array{Int,3}}
end


"""
    Volume <: Any

Volume, i.e. scanned structure of the brain, either tumor or organ-at-risk (OaR)

#### Fields
- `points :: Matrix{Float64} : ` 3-by-N matrix of the volume's scanned surface points
- "`ref   :: String          : ` reference id of the vector space, i.e. coordinate system (such as MRI, CT, ST, etc.), that the `points` belong to"

    Volume(vol_node::String)

Construct patient's `Volume` object from the data stored in his PatInfo.xml file.
`vol_node` must be a volume node found in PatInfo.xml file.
"""
struct Volume
    "`points::Matrix{Float64}`: 3-by-N matrix of the volume's scanned surface points"
    points::Matrix{Float64}
    "`ref::String`: reference id of the vector space, i.e. coordinate system (such as MRI, CT, ST, etc.), that the `points` belong to"
    ref::String

    function Volume(vol_node::String)
        regions = find(vol_node, "regions")
        # count total number of points in the volume
        num_points = mapreduce(region -> length(find(region, "vertices")), +, regions)
        # initialize points
        points = zeros(3, num_points)
        n = 0 # index counter
        # collect set of all points of the current volume looping over the slices (regions) along z axis
        for region ∈ regions
            vertices = find(region, "vertices")
            rng = (n+1:n+length(vertices))
            n = rng[end]
            for (i, vertex) ∈ zip(rng, vertices)
                points[:,i] = float( [ vertex["x"]; vertex["y"]; region["z"] ] )
            end
        end
        ### construct Volume
        new(points, vol_node["image_stack_ref",'*'])
    end
end


"""
    AffineMap <: Any

Affine map
``x ↦ \\textrm{matrix} ⋅ x + \\textrm{shift}``,
between pair of vector spaces, i.e. coordinate systems (such as MRI, CT, ST, etc.).

#### Fields
- `matrix :: Matrix{Float64} : ` linear part of the affine transformation
- `shift  :: Vector{Float64} : ` translational part of the affine transformation
- `from   :: String          : ` input vector space, i.e. coordinate system
- `to     :: String          : ` output vector space, i.e. coordinate system
"""
struct AffineMap
    "`matrix::Matrix{Float64}`: linear part of the affine transformation"
    matrix::Matrix{Float64}
    "`shift::Vector{Float64}`: translational part of the affine transformation"
    shift::Vector{Float64}
    "`from::String`: input vector space, i.e. coordinate system"
    from::String
    "`to::String`: output vector space, i.e. coordinate system"
    to::String
end
"""
AffineMap(registration::String)

Construct `AffineMap` object from the corresponding data stored in PatInfo.xml file.
`registration` must be a registration node found in PatInfo.xml file.
"""
function AffineMap(registration::String)
    # get transformation matrix R and translation vector t in the 3-by-4 block [R t]
    values = zeros(3,4)
    for i ∈ 0:2, j ∈ 0:3
        values[i+1,j+1] = float(registration["m$i$j"])
    end
    AffineMap(values[:,1:end-1], values[:,end], registration["stack_ref",'*'], registration["reference_stack_ref",'*'])
end


"""
    Patient <: Any

Patient's data collection.

#### Fields
- `isocenters :: Vector{Matrix{Float64}} : ` isocenters of treatment plan that was delivered to the patient
- `tumors     :: Vector{Volume}          : ` scanned surface points of tumor volumes
- `OaRs       :: Vector{Volume}          : ` scanned surface points of organ-at-risk volumes
- `maps       :: Vector{AffineMap}       : ` affine maps between vector spaces, i.e. coordinate systems of employed scans (MRI, CT, ST)

    Patient(patient_directory::String)

Construct `Patient` object from it's PatInfo.xml file.
`patient_directory` must be a path to the folder that contains PatInfo.xml file (extracted from `.lgp` patient file) and DICOM image files.
"""
struct Patient
    "`isocenters::Vector{Matrix{Float64}}`: isocenters of treatment plan that was delivered to the patient"
    isocenters::Vector{Matrix{Float64}}
    "`tumors::Vector{Volume}`: scanned surface points of tumor volumes"
    tumors::Vector{Volume}
    "`OaRs::Vector{Volume}`: scanned surface points of organ-at-risk volumes"
    OaRs::Vector{Volume}
    "`maps::Vector{AffineMap}`: affine maps between coordinate systems of employed scans (MRI, CT, Stereotactic)"
    maps::Vector{AffineMap}

    function Patient(patient_directory::String)

        # read PatInfo.xml file
        content = readPatInfo(patient_directory)

        # find approved treatment plan (i.e. with state = 5)
        plan = let
            plans = find(content, "patients/examinations/treatment_plans")
            k = 0
            for (i, plan) ∈ enumerate(plans)
                (plan["state"] == "5") && (k = i)
            end
            plans[k]
        end

        # get treatment targets (tumors) in the found plan
        targets = find(plan, "targets")

        # get all volumes of interest
        volumes = find(content, "patients/examinations/volumes")

        # initialize OaRs (organs-at-risk) & tumors
        tumors = Array{Volume}(length(targets))
        OaRs = Array{Volume}( length(volumes) - length(targets) )
        k = 0
        for (j, volume) ∈ enumerate(volumes)
            # check if the current volume is tumor
            if volume["classification"] == "2"
                k += 1
                tumors[k] = Volume(volume)
            else
                OaRs[j-k] = Volume(volume)
            end
        end

        # get affine transformations between vector spaces, i.e. coordinate systems that patient's volumes were registratered in
        registrations = find(content, "patients/examinations/registrations")

        # construct the patient object
        new(get_shots.(targets), tumors, OaRs, AffineMap.(registrations))
    end
end

"""
    (affmap::AffineMap)(x::Vector{R}) where {R <: Real}

Evaluate affine map `affmap` at a point `x`.
"""
(affmap::AffineMap)(x::Vector{R}) where {R <: Real} = (affmap.matrix * x + affmap.shift)::Vector{Float64}

"""
    readlgpfiles(dir::String) -> Vector{Patient}

Read `PatInfo.xml` file of every `.lgp` file present in the directory `dir`,
construct `Patient` object for each of them and return the result in array.
"""
function readlgpfiles(dir::String)

    # ensure that directory dir ends with '/'
    (dir[end] == '/') || (dir = dir * "/")

    # collect names all .lgp files in directory `dir`
    fileslgp, fileszip = let
        files = readdir(dir)
        islgp = [ file[end-3:end] == ".lgp" for file ∈ files ]
        files[islgp], [ dir * f[1:end-3] * "zip" for f ∈ files[islgp] ]
    end

    # preallocate patients array
    patients = Array{Patient}( length(fileslgp) )

    println("Processing files:")
    # for every .lgp file in directory `dir`
    for (i, (flgp, fzip)) ∈ enumerate(zip(fileslgp, fileszip))
        println(i, ". ", flgp)
        # rename it's extension to .zip file
        mv(dir * flgp, fzip)

        # extract PatInfo.xml from the resulting archive file,
        # overriding the previous one if present in destination
        zipfile[:ZipFile](fzip)[:extract]("PatInfo.xml", dir)

        # rename it's extension back to .lgp file
        mv(fzip, dir * flgp)

        # construct patient object from current PatInfo.xml file
        patients[i] = Patient(dir)
    end
    # remove remaining PatInfo.xml file
    rm(dir * "PatInfo.xml")

    return patients::Vector{Patient}
end


"""
A Julia module for handling `.lgp` files.
"""
LGP

end

from __future__ import (division, print_function, absolute_import)
import numpy as np
import shutil
from taskinit import mstool, tbtool
from split_cli import split_cli as split

clight = 2.99792458e+8          # [m/s] Speed of light

__all__ = ["export_uvtable", "uvtable"]

ms = mstool()
tb = tbtool()

def uvtable(root_file):
    '''
    This functions reads a root_file (the .ms extention is assumed) and create two uvtables 
    1- root_file_keepflagsFalse.txt (used for modeling)
    2- root_file_keepflagsTrue.txt  (used to create the ms of the residual and model)
    '''
    
    #Determines the number of channels (we assume that this is constant for all the spw)
    
    ms.open(root_file+'.ms')
    spw_info = ms.getspectralwindowinfo()
    kk=list(spw_info.keys())[0]
    #kk=str(sorted(map(int,spw_info.keys()))[0])
    Nchan = spw_info[kk]["NumChan"]
    npol = spw_info[kk]["NumCorr"]  
    ms.close()
    
    if (npol == 1):
        polcorr = 'single'
    elif(npol == 2):
        polcorr='dual'
    elif(npol ==4):
        polcorr='full'
    else:
        print('error, polcorr is not single, dual or full')

    u_noF, v_noF, Re_noF, Im_noF, Wei_noF = [], [], [], [], [] #All data is included
    u_F, v_F, Re_F, Im_F, Wei_F = [], [], [], [], []           #Only non-flagged data is included

    for j in range(Nchan):
        u, v, Re, Im, Wei, uf, vf, Ref, Imf, Weif = export_uvtable(tb, channel='all',polcorr=polcorr,split=split, split_args={'vis':root_file+'.ms','datacolumn':'DATA','spw':'*:'+str(j),'keepflags':True},verbose=True)
    
        u_F.append(u); v_F.append(v); Re_F.append(Re); Im_F.append(Im); Wei_F.append(Wei)
        u_noF.append(uf); v_noF.append(vf); Re_noF.append(Ref); Im_noF.append(Imf); Wei_noF.append(Weif)

    u_F   = np.hstack(u_F);    u_noF = np.hstack(u_noF)
    v_F   = np.hstack(v_F);    v_noF = np.hstack(v_noF)
    Re_F = np.hstack(Re_F);    Re_noF = np.hstack(Re_noF)
    Im_F  = np.hstack(Im_F);   Im_noF = np.hstack(Im_noF)
    Wei_F = np.hstack(Wei_F);  Wei_noF = np.hstack(Wei_noF)


    np.savetxt(root_file+'_keepflagsFalse.txt', np.column_stack([u_F, v_F, Re_F, Im_F, Wei_F]), fmt='%10.6e',
           delimiter='\t', header='u[lambda]\tv[lamda]\tRe(V)[Jy]\tIm(V)[Jy]\tweight')

    np.savetxt(root_file+'_keepflagsTrue.txt', np.column_stack([u_noF, v_noF, Re_noF, Im_noF, Wei_noF]), fmt='%10.6e',
           delimiter='\t', header='u[lambda]\tv[lamda]\tRe(V)[Jy]\tIm(V)[Jy]\tweight')


def export_uvtable(tb, vis="", split_args=None, split=None, channel='all',
                   polcorr='dual', fmt='%10.6e', datacolumn="MODEL_DATA", keep_tmp_ms=False,
                   verbose=True):
    """
    
    
    ------- THIS IS A MODIFIED VERSION OF THE ORIGINAL EXPORT UVTABLE BY JENNINGS -------
    
    
    
    Export visibilities from an MS Table to a uvtable. Requires execution inside CASA.

    Currently the only uvtable format supported is ASCII.

    Typicall call signature::

        export_uvtable('uvtable_new.txt', tb, channel='all', split=split,
                       split_args={'vis': 'sample.ms', 'datacolumn': 'DATA', 'spw':'0,1'},
                       verbose=True)

    Parameters
    ----------
    uvtable_filename : str
        Filename of the output uvtable, e.g. "uvtable.txt"
    tb : CASA `tb` object
        As tb parameter you **must** pass the tb object that is defined in the CASA shell.
        Since tb cannot be accessed outside CASA, export_uvtable('uvtable.txt', tb, ...)
        can be executed only inside CASA.
    vis : str, optional
        MS Table filename, e.g. mstable.ms
    split_args : dict, optional
        Default is None. If provided, perform a split before exporting the uvtable.
        The split_args dictionary is passed to the CASA::split task.
        The CASA::split task must be provided in input as split.
    split : optional
        CASA split task
    channel : str, optional
        If 'all', all channels are exported; if 'first' only the first channel of each spectral window (spw) is exported.
        Number of channels in each spw must be equal, otherwise you will get a CASA error, e.g.:
        `RuntimeError: ArrayColumn::getColumn cannot be done for column DATA; the array shapes vary: Table array conformance error`
        Default is 'all'.
    polcorr : str, optional
        If the MS Table contains dual polarisation data (polcorr='dual'), full polarisation (polcorr='full')
        or single polarisation (polcorr='single'). Default is dual.
    fmt : str, optional
        Format of the output ASCII uvtable.
    datacolumn: str, optional
        Data column to be extracted, e.g. "DATA", "CORRECTED_DATA", "MODEL_DATA".
    keep_tmp_ms : bool, optional
        If True, keeps the temporary outputvis created by the split command.
    verbose : bool, optional
        If True, print informative messages. Default: True


    Note
    ----
    By default, all the spws and all the channels of the `vis` MS table are exported.
    To export only the first channel in each spw, set channel='first'.

    To export only some spws provide split_args, e.g.::

        split_args = {'vis': 'input.ms', 'outputvis': 'input_tmp.ms', spw: '1,2'}
    
    Flagged data can be removed from the .ms by providing `split` and the `split_args` dictionary, e.g.::
    
        split_args = {'vis': 'input.ms', 'outputvis': 'input_tmp.ms', 'keepflags': False}

    If you try exporting visibilites from an MS table with spws with different number of channels,
    you will get a CASA error (see requirement for the `channel` parameter above).
    To avoid that, you can either use the CASA `split` task to create a new MS table with only spws
    with same number of channels. Alternatively (or additionally) you can channel average all the spws
    to the same number of channels.


    Example
    -------
    From within CASA, to extract all the visibilities from an MS table::

        export_uvtable('uvtable.txt', tb, vis='sample.ms', channel='all')

    where `tb` is the CASA tb object (to inspect it type `tb` in the CASA shell).
    For more information on `tb` see `<https://casa.nrao.edu/docs/CasaRef/table-Module.html>`_

    From within CASA, to extract the visibilities in spectral windows 0 and 2 use
    the `split_args` parameter and the CASA `split` task::

        export_uvtable('uvtable.txt', tb, channel='all', split=split,
         split_args={'vis': 'sample.ms' , 'datacolumn': 'DATA', 'spw':'0,2'})

    To perform these operations without running CASA interactively::

        casa --nologger --nogui -c "from uvplot import export_uvtable; export_uvtable(...)"

    ensuring that the strings are inside the '..' characters and not the "..." one.

    """
    if vis != "":
        MStb_name = vis

    if split_args:
        if split is None:
            raise RuntimeError("Missing split parameter: provide the CASA split object in input. "
                               "See typical call signature.")
        if vis != "" and vis != split_args['vis']:
            # raise RuntimeError("extract_uvtable: either provide `vis` or `split_args` as input parameters, not both.")
            raise RuntimeError(
                "extract_uvtable: vis={} input parameter doesn't match with split_args['vis']={}".format(
                    vis, split_args['vis']))

        if not 'outputvis' in split_args.keys():
            split_args.update(outputvis='mstable_tmp.ms')

        MStb_name = split_args['outputvis']

        if verbose:
            print("Applying split. Creating temporary MS table {} from original MS table {}".format
                  (MStb_name, split_args['vis']))

        split(**split_args)

        # after splitting, data is put into the "DATA" column of the new ms
        if datacolumn != 'DATA' and verbose:
            print('datacolumn has been changed from "{}" to "DATA" '
                  'in order to operate on the new ms'.format(datacolumn))

        datacolumn = "DATA"

    else:
        if vis == "":
            raise RuntimeError \
                ("Missing vis parameter: provide a valid MS table filename.")

    if verbose:
        print("Reading {}".format(MStb_name))

    tb.open(MStb_name)

    # get coordinates
    uvw = tb.getcol("UVW")
    u, v, w = [uvw[i, :] for i in range(3)]

    # get weights
    weights_orig = tb.getcol("WEIGHT")

    # get visibilities
    tb_columns = tb.colnames()
    
    # get flags
    flags = np.squeeze(tb.getcol("FLAG"))
    
    if datacolumn.upper() in tb_columns:
        data = tb.getcol(datacolumn)
    else:
        raise KeyError("datacolumn {} is not available.".format(datacolumn))

    spw = tb.getcol("DATA_DESC_ID")
    nspw = len(np.unique(spw))
    if nspw > 1:
        if split_args is None or \
                (split_args is not None and 'spw' not in split_args.keys()):
            print(
                "Warning: the MS table {} has {} spectral windows. By default all of them are exported."
                " To choose which spws to export, provide split_args with the spw parameter.".format(
                    MStb_name, nspw))

    # decide whether we export first channel, or all
    if channel == 'first':
        nchan = 1
        ich = 0
        if verbose:
            print("Exporting the first channel in each spw.")
    elif channel == 'all':
        nchan = data.shape[1]
        ich = slice(0, nchan)
        u = np.tile(u, nchan)
        v = np.tile(v, nchan)
        if verbose:
            print("Exporting {} channels per spw.".format(nchan))
    else:
        raise ValueError("Channel must be 'first' or 'all', not {}".format(channel))

    if polcorr == 'dual':
        # dual polarisation: extract the polarised visibilities and weights
        V_XX = data[0, ich, :].reshape(-1)
        V_YY = data[1, ich, :].reshape(-1)
        weights_XX = weights_orig[0, :]
        weights_YY = weights_orig[1, :]
        if nchan > 1:
            weights_XX = np.tile(weights_XX, nchan)
            weights_YY = np.tile(weights_YY, nchan)

        # compute weighted average of the visibilities and weights
        V = (V_XX * weights_XX + V_YY * weights_YY) / (weights_XX + weights_YY)
        weights = weights_XX + weights_YY
    elif polcorr == 'full':
        # full polarisation: extract the polarised visibilities and weights
        V_XX = data[0, ich, :].reshape(-1)
        V_YY = data[3, ich, :].reshape(-1)
        weights_XX = weights_orig[0, :]
        weights_YY = weights_orig[3, :]
        if nchan > 1:
            weights_XX = np.tile(weights_XX, nchan)
            weights_YY = np.tile(weights_YY, nchan)

        # compute weighted average of the visibilities and weights
        V = (V_XX * weights_XX + V_YY * weights_YY) / (weights_XX + weights_YY)
        weights = weights_XX + weights_YY
    elif polcorr == 'single':
        # single polarisation
        V = data[0, ich, :].reshape(-1)
        weights = weights_orig
        if nchan > 1:
            weights = np.tile(weights, nchan)

    spw_path = tb.getkeyword('SPECTRAL_WINDOW').split()[-1]
    tb.close()

    # get the mean observing frequency
    tb.open(spw_path)
    freqs = np.squeeze(tb.getcol('CHAN_FREQ'))  # [GHz]
    tb.close()

    # associate each datapoint with a frequency
    get_freq = lambda ispw: freqs[ispw]
    
    if (nspw > 1):
        freqs = get_freq(spw)
    if (nspw == 1 and nchan == 1):
        freqs = freqs
    if (nspw == 1 and nchan > 1):
        freqs = freqs[0]

    u, v = u*freqs / clight, v*freqs / clight
    if split_args:
        if not keep_tmp_ms:
            from subprocess import call
            if verbose:
                print("Removing temporary MS table {}".format(
                    split_args['outputvis']))
            call("rm -rf {}".format(split_args['outputvis']), shell=True)
            
    print ("flagged data not included!")
    flags = np.where(np.logical_not(flags[0])*np.logical_not(flags[1]) == True)
    data_u = u[flags]
    data_v = v[flags]
    data_real = V.real[flags]
    data_imag = V.imag[flags]
    data_wspec = weights[flags]
            
    return data_u, data_v, data_real, data_imag, data_wspec, u, v, V.real, V.imag, weights

#!/bin/bash
git-filter-repo \
--file-info-callback "
if filename.endswith(b'.ipynb'):
    print(f'\nProcessing {filename.decode()}')
    
    import copy
    
    try:
        import nbformat
        from nbstripout import strip_output
        
        # Get the file contents using the blob_id
        contents = value.get_contents_by_identifier(blob_id)

        nb = nbformat.reads(contents.decode('utf-8'), as_version=nbformat.NO_CONVERT)
        nb_original = copy.deepcopy(nb)

        # Customize parameters as needed:
        nb_stripped = strip_output(
            nb,
            keep_output=False,
            keep_count=False,
            keep_id=False,
            extra_keys=[
                'metadata.signature',
                'metadata.kernelspec',
                'metadata.widgets', 
                'cell.metadata.collapsed',
                'cell.metadata.ExecuteTime',
                'cell.metadata.execution',
                'cell.metadata.heading_collapsed',
                'cell.metadata.hidden',
                'cell.metadata.scrolled'
            ],
            drop_empty_cells=False,
            drop_tagged_cells=[],
            strip_init_cells=False,
            max_size=0
        )
        
        if nb_original != nb_stripped:
            # Convert cleaned notebook back to bytes
            new_contents = nbformat.writes(nb_stripped).encode('utf-8')
            
            print(f'  → Cleaned {filename.decode()}: {len(contents)} → {len(new_contents)} bytes')
            
            new_blob_id = value.insert_file_with_contents(new_contents)
            return (filename, mode, new_blob_id)
        else:
            print(f'  → No changes needed for {filename.decode()}')
    except Exception as e:
        print(f'Error processing {filename.decode()}: {e}')
        import traceback
        traceback.print_exc()
        # Return unchanged file on error
        return (filename, mode, blob_id)
# Return unchanged file if it's not a notebook
return (filename, mode, blob_id)
"
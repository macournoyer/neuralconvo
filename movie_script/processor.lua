local Processor = torch.class("e.MovieScript.Processor")

-- TODO use a visitor?

-- Cleans up and aggregate a movie script into a 1-to-1 dialog.
function Processor:visit(script)
  for i, entry in ipairs(script) do
    
  end
end

function Processor:visitDialog(script)

end
